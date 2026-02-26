import json
import vllm
from transformers import AutoTokenizer
import argparse
import os
import base64
from io import BytesIO
from PIL import Image

parser = argparse.ArgumentParser(description="使用vLLM生成模型答案")
parser.add_argument("--model", type=str, required=True, 
                    help="模型路径")
parser.add_argument("--input", type=str, required=True,
                    help="输入JSON文件（包含问题、答案、图像）")
parser.add_argument("--output", type=str, required=True,
                    help="输出JSON文件（包含model_answer）")
parser.add_argument("--gpu", type=int, default=0,
                    help="GPU索引")
parser.add_argument("--num_samples", type=int, default=1,
                    help="每个问题生成几个答案")
args = parser.parse_args()

print(f"[GPU {args.gpu}] ========================================")
print(f"[GPU {args.gpu}] 并行答案生成")
print(f"[GPU {args.gpu}] ========================================")
print(f"[GPU {args.gpu}] 模型: {args.model}")
print(f"[GPU {args.gpu}] 输入: {args.input}")
print(f"[GPU {args.gpu}] 输出: {args.output}")
print(f"[GPU {args.gpu}] GPU: {args.gpu}")
print(f"[GPU {args.gpu}] ========================================")

print(f"[GPU {args.gpu}] 加载数据...")
try:
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"[GPU {args.gpu}]    ✓ 加载了 {len(data)} 条数据")
except FileNotFoundError:
    print(f"[GPU {args.gpu}] 输入文件不存在")
    exit(1)
except Exception as e:
    print(f"[GPU {args.gpu}] 加载数据失败: {e}")
    exit(1)

print(f"[GPU {args.gpu}] 初始化vLLM模型...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.85,
        seed=args.gpu,
        trust_remote_code=True,
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None,
        n=args.num_samples,
    )
    
    print(f"[GPU {args.gpu}]    模型加载成功")
except Exception as e:
    print(f"[GPU {args.gpu}] 模型加载失败: {e}")
    exit(1)

print(f"[GPU {args.gpu}] 准备输入...")

def b64_to_image(b64_str, min_size=224):
    """将base64字符串转换为PIL图像，并确保尺寸满足模型要求"""
    try:
        if not b64_str or not isinstance(b64_str, str):
            return None
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        

        width, height = img.size
        if width < min_size or height < min_size:
            print(f"[GPU {args.gpu}]    图像尺寸太小 ({width}x{height})，调整到最小尺寸...")

            scale = max(min_size / width, min_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"[GPU {args.gpu}]    ✓ 已调整到 {new_width}x{new_height}")
        
        return img
    except Exception as e:
        print(f"[GPU {args.gpu}]    图像处理失败: {e}")
        return None


placeholder = "<|image_pad|>"
prompts = []
images_pil = []
valid_indices = []

for i, item in enumerate(data):
    question = item.get('question', '')
    image_b64 = item.get('image', '')
    
    if not question:
        print(f"[GPU {args.gpu}]    跳过空问题 (索引 {i})")
        continue
    

    img = b64_to_image(image_b64)
    if img is None:
        print(f"[GPU {args.gpu}]    跳过无效图像 (索引 {i})")
        continue
    

    width, height = img.size
    if width < 10 or height < 10:
        print(f"[GPU {args.gpu}]    跳过尺寸异常的图像 (索引 {i}, 尺寸: {width}x{height})")
        continue
    

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    prompts.append(prompt)
    images_pil.append(img)
    valid_indices.append(i)

print(f"[GPU {args.gpu}]    ✓ 准备了 {len(prompts)} 个有效输入")

if not prompts:
    print(f"[GPU {args.gpu}] 没有有效输入")
    with open(args.output, "w") as f:
        json.dump([], f)
    exit(0)


print(f"[GPU {args.gpu}] 🚀 开始生成...")

valid_chats = []
for prompt, img in zip(prompts, images_pil):
    valid_chat = {
        "prompt": prompt,
        "multi_modal_data": {"image": img}
    }
    valid_chats.append(valid_chat)

try:
    responses = model.generate(valid_chats, sampling_params=sample_params, use_tqdm=True)
    print(f"[GPU {args.gpu}]    ✓ 生成完成")
except Exception as e:
    print(f"[GPU {args.gpu}] 生成失败: {e}")
    import traceback
    traceback.print_exc()
    

    results = []
    for original_idx in valid_indices:
        item = data[original_idx].copy()
        item['model_answer'] = ""
        item['generation_error'] = True
        item['error_message'] = str(e)
        results.append(item)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[GPU {args.gpu}]    已保存错误结果到: {args.output}")
    exit(1)


print(f"[GPU {args.gpu}] 处理结果...")

results = []
for idx, response, original_idx in zip(range(len(responses)), responses, valid_indices):
    try:

        if args.num_samples == 1:
            model_answer = response.outputs[0].text if response.outputs else ""
        else:

            model_answer = response.outputs[0].text if response.outputs else ""
        

        item = data[original_idx].copy()
        item['model_answer'] = model_answer
        item['generation_error'] = False
        
        results.append(item)
        
        if (idx + 1) % 10 == 0:
            print(f"[GPU {args.gpu}]    处理进度: {idx + 1}/{len(responses)}")
    
    except Exception as e:
        print(f"[GPU {args.gpu}]    处理第 {idx} 个结果失败: {e}")
        item = data[original_idx].copy()
        item['model_answer'] = ""
        item['generation_error'] = True
        results.append(item)

print(f"[GPU {args.gpu}]    ✓ 处理了 {len(results)} 个结果")


print(f"[GPU {args.gpu}] 保存结果...")
try:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[GPU {args.gpu}]    ✓ 已保存到: {args.output}")
except Exception as e:
    print(f"[GPU {args.gpu}] 保存失败: {e}")
    exit(1)


success_count = sum(1 for r in results if not r.get('generation_error', False))
error_count = len(results) - success_count

print(f"[GPU {args.gpu}] ========================================")
print(f"[GPU {args.gpu}] 完成！")
print(f"[GPU {args.gpu}] ========================================")
print(f"[GPU {args.gpu}] 总数: {len(results)}")
print(f"[GPU {args.gpu}] 成功: {success_count}")
print(f"[GPU {args.gpu}] 失败: {error_count}")
print(f"[GPU {args.gpu}] ========================================")
