import json
import os
import sys
from collections import defaultdict
from pathlib import Path


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from weakness_analysis_by_imagetype import (
    aggregate_weaknesses_by_imagetype,
    generate_weakness_prompt
)


def regenerate_prompt_from_detailed_results(detailed_file: str, output_file: str = None):
    
    if not os.path.exists(detailed_file):
        print(f"文件不存在: {detailed_file}")
        return False
    

    results = []
    with open(detailed_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            results.append(data)
    
    print(f"✓ 读取了 {len(results)} 条结果")
    

    print(f"\n按图像类型聚合...")
    stats = aggregate_weaknesses_by_imagetype(results)
    

    print(f"\n图像类型统计:")
    for img_type, data in sorted(stats.items(), key=lambda x: x[1]['accuracy']):
        print(f"  {img_type:20s}: {data['accuracy']*100:5.1f}% "
              f"({data['correct']}/{data['total_evaluated']})")
        if data.get('error_type_counts'):
            for error_type, count in sorted(data['error_type_counts'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"      - {error_type}: {count}")
    

    print(f"\n生成弱点提示...")
    weakness_prompt = generate_weakness_prompt(stats)
    

    if output_file is None:
        output_file = os.path.join(os.path.dirname(detailed_file), 'weakness_prompt.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(weakness_prompt)
    
    print(f"\n弱点提示已保存到: {output_file}")
    print("="*80)
    

    print(f"\n内容预览:")
    print("-"*80)
    lines = weakness_prompt.split('\n')
    for line in lines[:30]:  # 显示前30行
        print(line)
    if len(lines) > 30:
        print(f"... (共 {len(lines)} 行)")
    print("-"*80)
    
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python regenerate_weakness_prompt.py <detailed_results.jsonl>")
        print("")
        print("示例:")
        print("  python regenerate_weakness_prompt.py weakness_analysis_iter1/stage2_analysis/detailed_results.jsonl")
        sys.exit(1)
    
    detailed_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = regenerate_prompt_from_detailed_results(detailed_file, output_file)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

