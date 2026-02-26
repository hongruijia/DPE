import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
from PIL import Image
import base64
from io import BytesIO


sys.path.append(str(Path(__file__).parent.parent))

from question_generate.api_clients import O3Client, Qwen3VLClient
from question_generate.image_type_strategies import ImageType, ImageTypeStrategy




IMAGE_TYPES = [
    "geometric",
    "medical", 
    "chart_graph",
    "text_dense",
    "diagram",
    "math_formula",
    "map",
    "natural_scene",
    "artistic",
    "everyday_object",
    "architectural",
    "mixed"
]


def identify_image_type_from_question(question: str, answer: str, image_desc: str = "") -> str:

    try:

        if '\nA.' in question or '\nB.' in question:
            q_type = 'multiple choice'
        else:
            q_type = 'numerical'
        

        img_type = ImageTypeStrategy.identify_image_type(
            plan_summary=question,
            question_type=q_type,
            original_question=question,
            use_gpt4o=True
        )
        return img_type.value
    except Exception as e:
        print(f"[identify_image_type_from_question] 错误: {e}")
        return "mixed"


def load_dataset(dataset_path: str, sample_size: int, random_seed: int = 42) -> List[Dict]:

    import glob
    
    print(f"加载数据集: {dataset_path}")
    

    if os.path.isdir(dataset_path):

        parquet_files = sorted(glob.glob(os.path.join(dataset_path, '*.parquet')))
        print(f"   找到 {len(parquet_files)} 个parquet文件")
        
        if not parquet_files:
            raise FileNotFoundError(f"目录中没有找到parquet文件: {dataset_path}")
        

        random.seed(random_seed)
        selected_file = random.choice(parquet_files)
        print(f"   随机选择: {os.path.basename(selected_file)}")
        
        df = pd.read_parquet(selected_file)
    else:

        df = pd.read_parquet(dataset_path)
    
    print(f"   数据量: {len(df)}")
    

    random.seed(random_seed)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed)
    
    print(f"   采样数量: {len(df)}")
    

    def convert_to_serializable(obj):
        import base64
        import numpy as np
        
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, dict):

            if 'bytes' in obj and isinstance(obj['bytes'], bytes):
                return base64.b64encode(obj['bytes']).decode('utf-8')
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        else:
            return obj
    

    samples = []
    for idx, row in df.iterrows():
        sample = {
            'id': str(idx),
            'question': str(row.get('problem', row.get('question', ''))),
            'answer': str(row.get('answer', '')),
            'image': convert_to_serializable(row.get('images', row.get('image', ''))),
            'problem_type': str(row.get('problem_type', 'unknown'))
        }
        samples.append(sample)
    
    return samples

def evaluate_answer_with_llm(question: str, correct_answer: str, model_answer: str, 
                             image_base64: str, judge_client: Qwen3VLClient) -> Dict:

    try:
        img_bytes = base64.b64decode(image_base64)
        image_pil = Image.open(BytesIO(img_bytes))
    except Exception as e:
        print(f" 图像解码失败: {e}")
        return {
            'is_correct': False,
            'error_type': 'image_error',
            'error_description': f'图像解码失败: {e}'
        }
    

    prompt = f"""You are an expert answer evaluator. Compare the model's answer with the correct answer.

Question: {question}

Correct Answer: {correct_answer}

Model Answer: {model_answer}

Task:
1. Determine if the model's answer is correct (consider equivalent expressions)
2. If incorrect, identify the error type and describe the weakness

Output JSON:
{{
    "is_correct": true/false,
    "error_type": "one of: visual_recognition_error, calculation_error, reasoning_error, misunderstanding, other",
    "error_description": "Brief description of what went wrong (if incorrect)"
}}

Important: 
- For numerical answers, consider small rounding differences as correct
- For multiple choice, only the letter matters
- Focus on whether the model understood the visual content correctly

Output ONLY the JSON, no extra text."""

    try:
        messages = [{"role": "user", "content": prompt}]
        


        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image_pil.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            response = judge_client.chat(messages, image_path=tmp_path, max_tokens=1024)
            

            os.unlink(tmp_path)
            

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                print(f"   评估响应无JSON: {response}")
                return {
                    'is_correct': False,
                    'error_type': 'evaluation_error',
                    'error_description': 'Judge返回格式错误'
                }
        except Exception as e:
            print(f"   评估异常: {e}")
            return {
                'is_correct': False,
                'error_type': 'evaluation_error',
                'error_description': f'评估失败: {e}'
            }
            
    except Exception as e:
        print(f"  评估出错: {e}")
        return {
            'is_correct': False,
            'error_type': 'evaluation_error',
            'error_description': f'评估异常: {e}'
        }


def aggregate_weaknesses_by_imagetype(results: List[Dict]) -> Dict:

    stats = {}
    
    for img_type in IMAGE_TYPES:
        stats[img_type] = {
            'total_evaluated': 0,
            'correct': 0,
            'accuracy': 0.0,
            'error_rate': 0.0,
            'errors': [],
            'question_types_breakdown': defaultdict(lambda: {'count': 0, 'correct': 0})
        }
    

    for result in results:
        img_type = result.get('image_type', 'mixed')
        q_type = result.get('problem_type', 'unknown')
        
        if img_type not in stats:
            img_type = 'mixed'
        
        stats[img_type]['total_evaluated'] += 1
        stats[img_type]['question_types_breakdown'][q_type]['count'] += 1
        
        if result['is_correct']:
            stats[img_type]['correct'] += 1
            stats[img_type]['question_types_breakdown'][q_type]['correct'] += 1
        else:
            stats[img_type]['errors'].append({
                'question': result['question'],
                'correct_answer': result['answer'],
                'model_answer': result['model_answer'],
                'error_type': result['error_type'],
                'error_description': result['error_description']
            })
    

    for img_type in stats:
        total = stats[img_type]['total_evaluated']
        if total > 0:
            stats[img_type]['accuracy'] = stats[img_type]['correct'] / total
            stats[img_type]['error_rate'] = 1 - stats[img_type]['accuracy']
        

        for q_type in stats[img_type]['question_types_breakdown']:
            breakdown = stats[img_type]['question_types_breakdown'][q_type]
            if breakdown['count'] > 0:
                breakdown['accuracy'] = breakdown['correct'] / breakdown['count']
    

    for img_type in stats:
        stats[img_type]['error_descriptions_by_type'] = extract_weaknesses_from_errors(
            stats[img_type]['errors']
        )

        del stats[img_type]['errors']
    

    stats = {k: v for k, v in stats.items() if v['total_evaluated'] > 0}
    
    return stats


def extract_weaknesses_from_errors(errors: List[Dict]) -> Dict[str, List[str]]:

    error_by_type = defaultdict(list)
    
    for error in errors:
        error_type = error['error_type']
        error_desc = error['error_description']

        error_by_type[error_type].append(error_desc[:200])
    
    return dict(error_by_type)


def calculate_generation_weights(stats: Dict, base_weight: float = 1.0) -> Dict:

    weights = {}
    
    for img_type, data in stats.items():
        accuracy = data['accuracy']
        
        if accuracy < 0.70:
            weight = 2.0
        elif accuracy < 0.75:
            weight = 1.5
        elif accuracy < 0.85:
            weight = 1.0
        elif accuracy < 0.90:
            weight = 0.8
        else:
            weight = 0.5
        
        weights[img_type] = weight
    

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    return weights


def analyze_weakness_with_llm(img_type: str, error_descriptions: List[str], error_type: str, qwen_client=None) -> Dict[str, str]:

    import re
    import json as json_module
    

    if qwen_client is None:
        from api_clients import Qwen3VLClient
        qwen_client = Qwen3VLClient(model_name="qwen-vl-max")
    

    sample_errors = error_descriptions[:5]  # 只取前5个作为示例
    errors_text = "\n".join([f"- {err}" for err in sample_errors])
    
    prompt = f"""You are analyzing errors from a vision-language model.

Image Type: {img_type}
Error Type: {error_type}
Error Examples:
{errors_text}

Task: Based on these errors, provide guidance for generating new training questions.

Output ONLY this JSON (no extra text):
{{
    "generation_direction": "Specific, actionable guidance on what kinds of questions to generate to fix this weakness (2-3 sentences)"
}}"""
    
    max_retries = 3
    for retry in range(max_retries):
        try:

            messages = [{"role": "user", "content": prompt}]
            response = qwen_client.chat(
                messages=messages,
                max_tokens=400
            )
            
            if response:

                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        analysis = json_module.loads(json_match.group())
                        if 'generation_direction' in analysis:
                            return analysis
                    except json_module.JSONDecodeError:
                        pass
            
            if retry < max_retries - 1:
                import time
                time.sleep(2)
                print(f"    [Qwen3-vl Analysis] Retry {retry + 1}/{max_retries}")
                continue
        except Exception as e:
            if retry < max_retries - 1:
                import time
                time.sleep(2)
                print(f"    [Qwen3-vl Analysis] Error: {e}, retry {retry + 1}/{max_retries}")
                continue
    

    print(f"    [Qwen3-vl Analysis] Failed, using fallback")
    return {
        'generation_direction': f'Generate {img_type} questions focusing on {error_type} scenarios with varying difficulty levels'
    }


def generate_weakness_prompt(stats: Dict, qwen_client=None) -> str:

    if qwen_client is None:
        from api_clients import Qwen3VLClient
        qwen_client = Qwen3VLClient(model_name="qwen-vl-max")
    
    lines = ["[Model Weakness Analysis - Targeted Generation Guidance]", ""]
    

    weak_types = [] 
    moderate_types = [] 
    strong_types = []   
    
    for img_type, data in stats.items():
        accuracy = data['accuracy']
        if accuracy < 0.70:
            weak_types.append((img_type, data))
        elif accuracy < 0.85:
            moderate_types.append((img_type, data))
        else:
            strong_types.append((img_type, data))
    

    weak_types.sort(key=lambda x: x[1]['accuracy'])
    moderate_types.sort(key=lambda x: x[1]['accuracy'])
    strong_types.sort(key=lambda x: x[1]['accuracy'])
    

    if weak_types:
        lines.append("=== HIGH PRIORITY: Types Requiring Significant Improvement ===")
        lines.append("")
        for img_type, data in weak_types:
            lines.append(f"[{img_type.upper()}] (Accuracy: {data['accuracy']*100:.1f}%, {data['total_evaluated']} samples)")
            

            if data['error_descriptions_by_type']:
                lines.append(f"  Weakness Analysis:")

                sorted_errors = sorted(
                    data['error_descriptions_by_type'].items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )
                
                for error_type, descriptions in sorted_errors:
                    count = len(descriptions)
                    total_errors = sum(len(v) for v in data['error_descriptions_by_type'].values())
                    pct = count / total_errors * 100 if total_errors > 0 else 0
                    
                    lines.append(f"    [{error_type}] ({count} cases, {pct:.0f}%)")
                    

                    print(f"  Analyzing {img_type} - {error_type} with Qwen3-vl...")
                    analysis = analyze_weakness_with_llm(img_type, descriptions, error_type, qwen_client)
                    
                    lines.append(f"      → {analysis['generation_direction']}")
            
            lines.append("")
    

    if moderate_types:
        lines.append("=== MODERATE PRIORITY: Types Needing Consolidation ===")
        lines.append("")
        for img_type, data in moderate_types:
            lines.append(f"[{img_type.upper()}] (Accuracy: {data['accuracy']*100:.1f}%, {data['total_evaluated']} samples)")
            

            if data['error_descriptions_by_type']:
                main_error_type = max(data['error_descriptions_by_type'].items(), key=lambda x: len(x[1]))
                error_type, descriptions = main_error_type
                count = len(descriptions)
                
                lines.append(f"  Primary Challenge: {error_type} ({count} cases)")
                

                print(f"  Analyzing {img_type} - {error_type} with Qwen3-vl...")
                analysis = analyze_weakness_with_llm(img_type, descriptions, error_type, qwen_client)
                lines.append(f"    → {analysis['generation_direction']}")
            
            lines.append("")
    

    if strong_types:
        lines.append("=== LOW PRIORITY: Well-Mastered Types ===")
        lines.append("")
        for img_type, data in strong_types:
            lines.append(f"[{img_type.upper()}] (Accuracy: {data['accuracy']*100:.1f}%, {data['total_evaluated']} samples)")
            lines.append(f"  → Reduce generation for this type, allocate resources to weaker areas")
            lines.append("")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="按图像类型进行弱点分析")
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--dataset', type=str,
                        default='path_to_Vision-SR1-47K',
                        help='评估数据集路径')
    parser.add_argument('--output', type=str, default='./weakness_analysis',
                        help='输出目录')
    parser.add_argument('--sample-size', type=int, default=200,
                        help='采样数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    run_weakness_analysis(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        sample_size=args.sample_size,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()

