import json
import vllm
from transformers import AutoTokenizer
import argparse
import os
import stopit
from mathruler.grader import extract_boxed_content, grade_answer
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description="为生成的问题评分（自采样一致性）")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                    help="模型路径")
parser.add_argument("--input_file", type=str, required=True,
                    help="输入JSON文件（包含 question, answer, image）")
parser.add_argument("--output_file", type=str, required=True,
                    help="输出JSON文件（添加 score 字段）")
parser.add_argument("--num_samples", type=int, default=9, 
                    help="每个问题生成的候选答案数")
parser.add_argument("--suffix", type=str, default="0", 
                    help="进程标识（用于日志）")
parser.add_argument("--skip_existing", action="store_true",
                    help="跳过已经有 score 的数据")
args = parser.parse_args()


@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    return grade_answer(res1, res2)

def decode_base64_image(b64_str):
    try:
        if not b64_str:
            return None
        img_bytes = base64.b64decode(b64_str)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"[{args.suffix}] 图片解码失败: {e}")
        return None

def compare_with_golden(generated_results, golden_answer, timeout=10):

    if not generated_results:
        return 0, 0
    

    golden_boxed = golden_answer.strip()
    
    consistent_count = 0
    valid_count = 0
    
    for result in generated_results:
        if not result:
            continue
        
        valid_count += 1
        

        if result == golden_boxed:
            consistent_count += 1
            continue
        

        if 'no ' in result.lower() and 'no ' in golden_boxed.lower():
            consistent_count += 1
            continue
        

        try:
            match_1 = grade_answer_with_timeout(result, golden_boxed, timeout=timeout)
            if match_1 == 'TIMED_OUT':
                print(f"[{args.suffix}] 比较超时: '{result[:30]}...' vs golden")
                continue
            
            if match_1:
                consistent_count += 1
                continue
            

            match_2 = grade_answer_with_timeout(golden_boxed, result, timeout=timeout)
            if match_2 == 'TIMED_OUT':
                print(f"[{args.suffix}] 反向比较超时")
                continue
            
            if match_2:
                consistent_count += 1
                
        except Exception as e:
            print(f"[{args.suffix}] 比较异常: {e}")
            continue
    
    return consistent_count, valid_count


def main():

    print(f"[{args.suffix}] 加载数据: {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    print(f"[{args.suffix}] 总计: {len(data)} 条数据")
    

    if args.skip_existing:
        to_score = [item for item in data if 'score' not in item or item['score'] == 1.0]
        print(f"[{args.suffix}] 需要评分: {len(to_score)} 条（跳过已有score的）")
    else:
        to_score = data
        print(f"[{args.suffix}] 全部重新评分")
    
    if not to_score:
        print(f"[{args.suffix}] 所有数据已评分，退出")
        return
    

    questions = []
    golden_answers = []
    images_base64 = []
    question_types = []
    indices = [] 
    
    for idx, item in enumerate(data):
        if args.skip_existing and 'score' in item and item['score'] != 1.0:
            continue
        
        q = item.get('question', '')
        a = item.get('answer', '')
        img = item.get('image', '')
        qt = item.get('question_type', 'unknown')
        
        if not q or not a or not img:
            print(f"[{args.suffix}] 跳过不完整数据: index={idx}")
            continue
        
        questions.append(q)
        golden_answers.append(a)
        images_base64.append(img)
        question_types.append(qt)
        indices.append(idx)
    
    if not questions:
        print(f"[{args.suffix}] 没有有效数据需要评分")
        return
    
    print(f"[{args.suffix}] 有效数据: {len(questions)} 条")
    

    print(f"[{args.suffix}] 初始化模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.85,
        seed=int(args.suffix) if args.suffix.isdigit() else 42,
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.num_samples,  # 生成多个候选答案
    )
    

    print(f"[{args.suffix}] 构建prompts...")
    placeholder = "<|image_pad|>"
    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for q in questions
    ]
    

    print(f"[{args.suffix}] 解码图片...")
    images_pil = [decode_base64_image(b64) for b64 in images_base64]
    

    valid_chats = []
    valid_indices = []
    for i, (prompt, img) in enumerate(zip(prompts, images_pil)):
        if img is not None:
            valid_chats.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img}
            })
            valid_indices.append(i)
        else:
            print(f"[{args.suffix}] 图片解码失败，跳过: index={indices[i]}")
    
    print(f"[{args.suffix}] 有效输入: {len(valid_chats)} 条")
    

    print(f"[{args.suffix}] 开始生成（每题{args.num_samples}个候选）...")
    responses = model.generate(valid_chats, sampling_params=sample_params, use_tqdm=True)
    print(f"[{args.suffix}] 生成完成")
    

    print(f"[{args.suffix}] 计算一致性得分...")
    scored_count = 0
    
    for i, response in enumerate(tqdm(responses, desc=f"[{args.suffix}] 评分")):
        original_idx = indices[valid_indices[i]]
        golden_answer = golden_answers[valid_indices[i]]
        
        try:

            generated_results = [
                extract_boxed_content(output.text) 
                for output in response.outputs
            ]
            generated_results = [res for res in generated_results if res]
            
            if not generated_results:
                print(f"[{args.suffix}] 未提取到答案: index={original_idx}")
                data[original_idx]['score'] = 0.0
                continue
            

            consistent_count, valid_count = compare_with_golden(
                generated_results, 
                golden_answer,
                timeout=10
            )
            
            if valid_count == 0:
                score = 0.0
            else:
                score = consistent_count / valid_count
            

            data[original_idx]['score'] = score
            data[original_idx]['consistency_info'] = {
                'consistent_count': consistent_count,
                'total_candidates': valid_count,
                'all_candidates': generated_results
            }
            
            scored_count += 1
            
            if i % 10 == 0:
                print(f"[{args.suffix}] 进度: {i}/{len(responses)}, "
                      f"当前score={score:.2f} ({consistent_count}/{valid_count})")
            
        except Exception as e:
            print(f"[{args.suffix}] 处理失败 index={original_idx}: {e}")
            data[original_idx]['score'] = 0.0
            continue
    

    print(f"[{args.suffix}] 保存结果: {args.output_file}")
    

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

    scores = [item.get('score', 0.0) for item in data if 'score' in item]
    
    print(f"\n[{args.suffix}] ========================================")
    print(f"[{args.suffix}] 评分完成！")
    print(f"[{args.suffix}] ========================================")
    print(f"[{args.suffix}] 评分数量: {scored_count}")
    print(f"[{args.suffix}] 平均得分: {sum(scores)/len(scores):.3f}")
    print(f"[{args.suffix}] 最高得分: {max(scores):.3f}")
    print(f"[{args.suffix}] 最低得分: {min(scores):.3f}")
    print(f"[{args.suffix}] 输出文件: {args.output_file}")
    print(f"[{args.suffix}] ========================================")

if __name__ == '__main__':
    main()

