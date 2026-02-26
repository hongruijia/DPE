import os
import sys
import json
import argparse
import time
import threading
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


sys.path.append(str(Path(__file__).parent.parent))

from question_generate.api_clients import Qwen3VLClient
from question_generate.image_type_strategies import ImageType, ImageTypeStrategy
from question_evaluate.weakness_analysis_by_imagetype import (
    identify_image_type_from_question,
    evaluate_answer_with_llm,
    aggregate_weaknesses_by_imagetype,
    calculate_generation_weights,
    generate_weakness_prompt
)



class RateLimiter:
    
    def __init__(self, min_interval: float = 2.0):
        self.min_interval = min_interval
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait(self):
        
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()


def evaluate_single_item(
    item: Dict,
    index: int,
    total: int,
    judge_client: Qwen3VLClient,
    rate_limiter: RateLimiter
) -> Dict:

    try:

        img_type = identify_image_type_from_question(
            item['question'],
            item['answer']
        )
        

        rate_limiter.wait()
        

        eval_result = evaluate_answer_with_llm(
            item['question'],
            item['answer'],
            item['model_answer'],
            item['image'],
            judge_client
        )
        

        result = {
            'id': item.get('id', f'sample_{index}'),
            'question': item['question'],
            'answer': item['answer'],
            'model_answer': item['model_answer'],
            'image_type': img_type,
            'problem_type': item.get('problem_type', 'unknown'),
            'is_correct': eval_result['is_correct'],
            'error_type': eval_result.get('error_type', 'unknown'),
            'error_description': eval_result.get('error_description', '')
        }
        
        status = '正确' if result['is_correct'] else '错误'
        print(f"[{index}/{total}] {img_type:15s} {status}")
        
        return result
    
    except Exception as e:
        print(f"[{index}/{total}] 评估失败: {e}")
        return {
            'id': item.get('id', f'sample_{index}'),
            'question': item['question'],
            'answer': item['answer'],
            'model_answer': item.get('model_answer', ''),
            'image_type': 'UNKNOWN',
            'problem_type': item.get('problem_type', 'unknown'),
            'is_correct': False,
            'error_type': 'evaluation_error',
            'error_description': str(e)
        }


def analyze_answers_and_generate_report(
    answers_file: str,
    output_dir: str,
    max_workers: int = 3,
    min_interval: float = 2.0
):

    print("="*80)
    print("阶段2: 分析弱点并生成报告（并发模式）")
    print("="*80)
    print(f"输入文件: {answers_file}")
    print(f"输出目录: {output_dir}")
    print(f"最大并发: {max_workers} workers")
    print(f"最小间隔: {min_interval} 秒")
    print("="*80)
    print()
    

    os.makedirs(output_dir, exist_ok=True)
    

    print("加载答案数据...")
    with open(answers_file, 'r') as f:
        data = json.load(f)
    
    print(f"   总数据量: {len(data)}")
    

    valid_data = [item for item in data if not item.get('generation_error', False)]
    print(f"   有效数据: {len(valid_data)} 条")
    

    print(f"\n初始化评估模型（qwen-vl-max）...")
    judge_client = Qwen3VLClient(model_name="qwen-vl-max")
    rate_limiter = RateLimiter(min_interval=min_interval)
    

    print(f"\n开始并发评估（{max_workers} workers）...")
    print(f"预计时间: ~{len(valid_data) * min_interval / max_workers / 60:.1f} 分钟")
    print()
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(
                evaluate_single_item,
                item,
                i,
                len(valid_data),
                judge_client,
                rate_limiter
            ): i
            for i, item in enumerate(valid_data, 1)
        }
        

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                print(f"[{idx}] 任务执行失败: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\n评估完成，耗时: {elapsed_time/60:.1f} 分钟")
    

    print(f"\n聚合弱点统计...")
    stats = aggregate_weaknesses_by_imagetype(results)
    

    print(f"\n计算生成权重...")
    weights = calculate_generation_weights(stats)
    

    print(f"\n生成弱点提示（使用 Qwen3-vl 分析深层原因）...")
    print(f"   注意：这将对每个错误类型调用 Qwen3-vl 进行深度分析")
    print(f"   预计需要额外 1-3 分钟...")
    weakness_prompt = generate_weakness_prompt(stats, qwen_client=judge_client)
    

    print(f"\n保存结果...")
    

    with open(os.path.join(output_dir, 'detailed_results.jsonl'), 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    

    with open(os.path.join(output_dir, 'weakness_analysis.json'), 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    

    with open(os.path.join(output_dir, 'generation_weights.json'), 'w') as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    

    with open(os.path.join(output_dir, 'weakness_prompt.txt'), 'w') as f:
        f.write(weakness_prompt)
    

    summary = {
        'total_evaluated': len(results),
        'total_correct': sum(1 for r in results if r['is_correct']),
        'overall_accuracy': sum(1 for r in results if r['is_correct']) / len(results) if results else 0,
        'by_image_type': {
            img_type: {
                'total': data['total_evaluated'],
                'accuracy': data['accuracy'],
                'generation_weight': weights.get(img_type, 1.0)
            }
            for img_type, data in stats.items()
        }
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    

    print("\n" + "="*80)
    print("弱点分析完成！")
    print("="*80)
    print(f"总评估数: {summary['total_evaluated']}")
    print(f"总体准确率: {summary['overall_accuracy']*100:.2f}%")
    print(f"\n各图像类型准确率：")
    
    sorted_types = sorted(
        summary['by_image_type'].items(),
        key=lambda x: x[1]['accuracy']
    )
    
    for img_type, data in sorted_types:
        print(f"  {img_type:20s}: {data['accuracy']*100:5.1f}% "
              f"(样本数: {data['total']:3d}, 权重: {data['generation_weight']:.2f})")
    
    print(f"\n输出文件：")
    print(f"  - {output_dir}/weakness_analysis.json")
    print(f"  - {output_dir}/generation_weights.json")
    print(f"  - {output_dir}/weakness_prompt.txt")
    print(f"  - {output_dir}/summary.json")
    print(f"  - {output_dir}/detailed_results.jsonl")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="阶段2：分析弱点（支持并发）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--answers', type=str, required=True,
                        help='阶段1生成的答案文件')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--max-workers', type=int, default=3,
                        help='最大并发worker数量（推荐2-5）')
    parser.add_argument('--min-interval', type=float, default=2.0,
                        help='API调用最小间隔（秒）')
    
    args = parser.parse_args()
    
    analyze_answers_and_generate_report(
        answers_file=args.answers,
        output_dir=args.output,
        max_workers=args.max_workers,
        min_interval=args.min_interval
    )


if __name__ == '__main__':
    main()

