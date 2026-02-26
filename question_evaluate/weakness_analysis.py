import os
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class WeaknessAnalysisTable:
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.weaknesses = self._load()
    
    def _load(self) -> Dict[str, List[Dict]]:
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.weaknesses, f, ensure_ascii=False, indent=2)
        print(f"弱点分析表已保存: {self.save_path}")
    
    def add_weakness(self, question_type: str, weakness_info: Dict[str, Any]):
        if question_type not in self.weaknesses:
            self.weaknesses[question_type] = []
        

        weakness_desc = weakness_info.get('weakness_description', '')
        for w in self.weaknesses[question_type]:
            if w.get('weakness_description') == weakness_desc:

                w['count'] = w.get('count', 1) + 1
                w['examples'].append({
                    'question_id': weakness_info.get('question_id'),
                    'error_type': weakness_info.get('error_type'),
                    'timestamp': weakness_info.get('timestamp')
                })
                return
        

        weakness_info['count'] = 1
        weakness_info['examples'] = [{
            'question_id': weakness_info.get('question_id'),
            'error_type': weakness_info.get('error_type'),
            'timestamp': weakness_info.get('timestamp')
        }]
        self.weaknesses[question_type].append(weakness_info)
    
    def get_weaknesses_by_type(self, question_type: str) -> List[Dict]:
        return self.weaknesses.get(question_type, [])
    
    def remove_weakness(self, question_type: str, weakness_id: int):
        if question_type in self.weaknesses:
            if 0 <= weakness_id < len(self.weaknesses[question_type]):
                removed = self.weaknesses[question_type].pop(weakness_id)
                print(f"✓ 已删除弱点: {removed.get('weakness_description')}")
    
    def update_weakness(self, question_type: str, weakness_id: int, new_info: Dict):
        if question_type in self.weaknesses:
            if 0 <= weakness_id < len(self.weaknesses[question_type]):
                self.weaknesses[question_type][weakness_id].update(new_info)
                print(f"✓ 已更新弱点信息")
    
    def get_summary(self) -> Dict[str, Any]:
        summary = {
            'total_types': len(self.weaknesses),
            'by_type': {}
        }
        for q_type, weaknesses in self.weaknesses.items():
            summary['by_type'][q_type] = {
                'weakness_count': len(weaknesses),
                'total_errors': sum(w.get('count', 0) for w in weaknesses),
                'top_weaknesses': sorted(
                    weaknesses, 
                    key=lambda x: x.get('count', 0), 
                    reverse=True
                )[:3]
            }
        return summary
    
    def print_summary(self):
        
        summary = self.get_summary()
        print(f"总问题类型数: {summary['total_types']}")
        
        for q_type, info in summary['by_type'].items():
            print(f"\n【{q_type}】")
            print(f"  - 弱点种类: {info['weakness_count']}")
            print(f"  - 错误总数: {info['total_errors']}")
            print(f"  - 主要弱点:")
            for i, w in enumerate(info['top_weaknesses'], 1):
                print(f"    {i}. {w.get('weakness_description', 'N/A')} (出现{w.get('count', 0)}次)")
        print("="*80)
    
    def generate_weakness_prompt_string(self, top_n: int = 5, min_count: int = 2) -> str:

        if not self.weaknesses:
            return "暂无发现明显弱点。"
        
        lines = []
        lines.append("【模型当前存在的主要弱点】")
        lines.append("")
        
        for q_type, weaknesses in self.weaknesses.items():

            sorted_weaknesses = sorted(
                weaknesses, 
                key=lambda x: x.get('count', 0), 
                reverse=True
            )
            

            filtered = [w for w in sorted_weaknesses if w.get('count', 0) >= min_count]
            
            if not filtered:
                continue
            
            lines.append(f"【{q_type}】")
            

            for i, w in enumerate(filtered[:top_n], 1):
                desc = w.get('weakness_description', '未知弱点')
                count = w.get('count', 0)
                error_type = w.get('error_type', '未分类')
                lines.append(f"{i}. {desc} [错误类型: {error_type}, 出现{count}次]")
            
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    def export_weakness_prompt_file(self, output_path: str, top_n: int = 5, min_count: int = 2):
        prompt_str = self.generate_weakness_prompt_string(top_n, min_count)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt_str)
        
        print(f"✓ 弱点提示字符串已保存: {output_path}")


class QwenVLMaxEvaluator:
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = "qwen-vl-max"
    
    def evaluate(self, question: str, ground_truth: str, model_answer: str, 
                 question_type: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
        

        evaluation_prompt = f"""你是一个专业的数学和视觉问答评估专家。请评估模型的答案质量。

**问题类型**: {question_type}

**问题**: {question}

**标准答案**: {ground_truth}

**模型回答**: {model_answer}

请从以下维度评估:
1. **正确性**: 答案是否正确（完全正确/部分正确/完全错误）
2. **错误类型**（如果错误）: 
   - 计算错误
   - 理解错误
   - 视觉识别错误
   - 推理错误
   - 格式错误
   - 其他
3. **弱点描述**: 如果答案有问题，简洁描述模型在这类问题上暴露的弱点
4. **评分**: 0-1之间的分数

请以JSON格式返回评估结果:
{{
    "is_correct": true/false,
    "score": 0.0-1.0,
    "error_type": "错误类型（如果错误）",
    "weakness_description": "弱点描述（如果有）",
    "feedback": "详细反馈"
}}
"""
        

        messages = [{"role": "user", "content": [{"type": "text", "text": evaluation_prompt}]}]
        

        if image_base64:
            messages[0]["content"].insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            


            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            print(f"评估失败: {e}")
            return {
                'is_correct': False,
                'score': 0.0,
                'error_type': 'evaluation_error',
                'weakness_description': f'评估失败: {str(e)}',
                'feedback': str(e)
            }


class ModelInference:
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from qwen_vl_utils import process_vision_info
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True
        )
        
        self.process_vision_info = process_vision_info
        print(f"模型加载完成")
    
    def generate_answer(self, question: str, image_base64: str) -> str:
        import torch
        

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}"
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
        

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        inputs = self.tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        

        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text


def load_dataset(dataset_path: str, sample_size: int = 200) -> List[Dict]:
    
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif dataset_path.endswith('.jsonl'):
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"不支持的文件格式: {dataset_path}")
    
    print(f"总数据量: {len(data)}")
    

    if len(data) > sample_size:
        sampled_data = random.sample(data, sample_size)
        print(f"随机抽取: {sample_size} 条")
    else:
        sampled_data = data
        print(f"使用全部数据: {len(data)} 条")
    
    return sampled_data


def run_weakness_analysis(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    sample_size: int = 200,
    resume: bool = True
):    

    os.makedirs(output_dir, exist_ok=True)
    

    weakness_table_path = os.path.join(output_dir, "weakness_analysis.json")
    weakness_table = WeaknessAnalysisTable(weakness_table_path)
    

    detailed_results_path = os.path.join(output_dir, "detailed_results.jsonl")
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    

    dataset = load_dataset(dataset_path, sample_size)
    

    processed_ids = set()
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            processed_ids = set(checkpoint.get('processed_ids', []))
        print(f"断点续传: 已处理 {len(processed_ids)} 条\n")
    

    print("初始化模型...")
    model_inference = ModelInference(model_path)
    evaluator = QwenVLMaxEvaluator()
    

    results = []
    correct_count = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc="处理样本")):

        sample_id = sample.get('id', f"sample_{idx}")
        
        if sample_id in processed_ids:
            continue
        

        question = sample.get('problem') or sample.get('question', '')
        ground_truth = sample.get('answer', '')
        image_base64 = sample.get('images') or sample.get('image', '')
        question_type = sample.get('problem_type') or sample.get('question_type', 'unknown')
        
        try:

            model_answer = model_inference.generate_answer(question, image_base64)
            

            evaluation = evaluator.evaluate(
                question=question,
                ground_truth=ground_truth,
                model_answer=model_answer,
                question_type=question_type,
                image_base64=image_base64
            )
            

            result = {
                'id': sample_id,
                'question': question,
                'question_type': question_type,
                'ground_truth': ground_truth,
                'model_answer': model_answer,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            

            if not evaluation.get('is_correct', False):
                weakness_info = {
                    'question_id': sample_id,
                    'weakness_description': evaluation.get('weakness_description', '未知弱点'),
                    'error_type': evaluation.get('error_type', '未知错误'),
                    'score': evaluation.get('score', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
                weakness_table.add_weakness(question_type, weakness_info)
            else:
                correct_count += 1
            

            with open(detailed_results_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            processed_ids.add(sample_id)
            

            if len(processed_ids) % 10 == 0:
                weakness_table.save()
                with open(checkpoint_path, 'w') as f:
                    json.dump({'processed_ids': list(processed_ids)}, f)
        
        except Exception as e:
            print(f"\n处理样本 {sample_id} 时出错: {e}")
            continue
    

    weakness_table.save()
    

    accuracy = correct_count / len(processed_ids) if processed_ids else 0
    final_stats = {
        'total_samples': len(processed_ids),
        'correct': correct_count,
        'accuracy': accuracy,
        'weakness_summary': weakness_table.get_summary()
    }
    
    stats_path = os.path.join(output_dir, "final_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    

    weakness_prompt_path = os.path.join(output_dir, "weakness_prompt.txt")
    weakness_table.export_weakness_prompt_file(weakness_prompt_path, top_n=5, min_count=2)
    

    print(f"\n分析完成！")
    print(f"总样本: {len(processed_ids)}")
    print(f"正确: {correct_count}")
    print(f"准确率: {accuracy*100:.2f}%")
    
    weakness_table.print_summary()
    

    print(f"\n" + "="*80)
    print("弱点提示字符串（可直接用于问题生成）")
    print("="*80)
    print(weakness_table.generate_weakness_prompt_string(top_n=5, min_count=2))
    print("="*80)
    
    print(f"\n输出文件:")
    print(f"  - 弱点分析表: {weakness_table_path}")
    print(f"  - 详细结果: {detailed_results_path}")
    print(f"  - 统计信息: {stats_path}")
    print(f"  - 弱点提示字符串: {weakness_prompt_path}  (用于下一轮问题生成)")


def main():
    parser = argparse.ArgumentParser(description="模型弱点诊断系统")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='数据集路径（JSON/JSONL/Parquet）')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（HuggingFace格式）')
    parser.add_argument('--output_dir', type=str, default='weakness_analysis_output',
                        help='输出目录')
    parser.add_argument('--sample_size', type=int, default=200,
                        help='抽样数量')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='不使用断点续传')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    

    random.seed(args.seed)
    
    run_weakness_analysis(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        resume=args.resume
    )


if __name__ == '__main__':
    main()

