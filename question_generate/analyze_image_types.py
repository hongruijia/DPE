#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from image_type_strategies import (
    ImageTypeStrategy, 
    ImageType,
    identify_image_type_with_gpt4o
)


def load_dataset(data_path: str, num_samples: int = 1000) -> List[Dict]:
    
    print(f"📂 加载数据集: {data_path}")
    print(f"📊 目标样本数: {num_samples}")
    
    samples = []
    
    data_path_obj = Path(data_path)
    
    if data_path_obj.is_dir():
        parquet_files = sorted(data_path_obj.glob("train-*.parquet"))
        print(f"   找到 {len(parquet_files)} 个parquet文件")
        
        for parquet_file in parquet_files:
            if len(samples) >= num_samples:
                break
                
            print(f"   读取: {parquet_file.name}")
            df = pd.read_parquet(parquet_file)
            
            for _, row in df.iterrows():
                if len(samples) >= num_samples:
                    break
                samples.append(row.to_dict())
                
    elif data_path_obj.suffix == ".parquet":
        df = pd.read_parquet(data_path)
        samples = df.head(num_samples).to_dict('records')
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")
    
    print(f"✓ 成功加载 {len(samples)} 个样本\n")
    return samples[:num_samples]


def extract_sample_info(sample: Dict) -> Dict[str, str]:

    question = ""
    for field in ["problem", "question", "conversations", "text", "prompt"]:
        if field in sample:
            if isinstance(sample[field], list):
                for conv in sample[field]:
                    if isinstance(conv, dict) and conv.get("from") == "human":
                        question = conv.get("value", "")
                        break
            else:
                question = str(sample[field])
            if question:
                break
    
    plan_summary = sample.get("plan_summary", question)

    question_type = sample.get("problem_type", sample.get("question_type", "unknown"))
    
    data_source = sample.get("data_source", "")
    if data_source:
        plan_summary = f"[{data_source}] {plan_summary}"
    
    return {
        "plan_summary": plan_summary,
        "question_type": question_type,
        "original_question": question
    }


def classify_samples(samples: List[Dict], use_gpt4o: bool = True) -> List[Dict]:

    print("🔍 开始对样本进行图片类型分类...")
    print(f"   分类方法: {'GPT-4o API' if use_gpt4o else '关键词匹配'}\n")
    
    classified_samples = []
    
    for idx, sample in enumerate(tqdm(samples, desc="分类进度")):
        info = extract_sample_info(sample)
        
        try:
            if use_gpt4o:
                image_type_str = identify_image_type_with_gpt4o(
                    plan_summary=info["plan_summary"],
                    question_type=info["question_type"],
                    original_question=info["original_question"]
                )
                try:
                    image_type = ImageType(image_type_str)
                except ValueError:
                    print(f"[样本 {idx}] GPT-4o返回无效类型: {image_type_str}, 使用关键词匹配")
                    image_type = ImageTypeStrategy.identify_image_type(
                        plan_summary=info["plan_summary"],
                        question_type=info["question_type"],
                        original_question=info["original_question"],
                        use_gpt4o=False
                    )
            else:
                image_type = ImageTypeStrategy.identify_image_type(
                    plan_summary=info["plan_summary"],
                    question_type=info["question_type"],
                    original_question=info["original_question"],
                    use_gpt4o=False
                )
            
            classified_samples.append({
                "index": idx,
                "image_type": image_type.value,
                "plan_summary": info["plan_summary"][:100],  # 截断以节省空间
                "question_type": info["question_type"],
                "original_question": info["original_question"][:100]  # 截断以节省空间
            })
            
        except Exception as e:
            print(f"\n[样本 {idx}] 分类失败: {e}")
            classified_samples.append({
                "index": idx,
                "image_type": "mixed",  # 默认为混合类型
                "plan_summary": info["plan_summary"][:100],
                "question_type": info["question_type"],
                "original_question": info["original_question"][:100],
                "error": str(e)
            })
    
    print("\n✓ 分类完成\n")
    return classified_samples


def generate_statistics(classified_samples: List[Dict]) -> Dict:

    print("📊 生成统计报告...\n")
    
    type_counts = Counter([s["image_type"] for s in classified_samples])
    total = len(classified_samples)
    
    statistics = {
        "total_samples": total,
        "type_distribution": {},
        "type_counts": dict(type_counts)
    }
    
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("="*80)
    print("图片类型分布统计")
    print("="*80)
    print(f"总样本数: {total}\n")
    print(f"{'类型':<25} {'数量':<10} {'比例':<10} {'描述'}")
    print("-"*80)
    
    for img_type, count in sorted_types:
        percentage = (count / total) * 100
        statistics["type_distribution"][img_type] = {
            "count": count,
            "percentage": round(percentage, 2)
        }
        
        try:
            type_enum = ImageType(img_type)
            strategy = ImageTypeStrategy.get_strategy(type_enum)
            description = strategy.get("description", "")
        except:
            description = ""
        
        print(f"{img_type:<25} {count:<10} {percentage:>6.2f}%   {description}")
    
    print("="*80)
    print()
    
    return statistics


def save_results(classified_samples: List[Dict], statistics: Dict, output_dir: str):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    detail_file = output_path / "image_type_classification_details.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(classified_samples, f, ensure_ascii=False, indent=2)
    print(f"✓ 详细分类结果已保存: {detail_file}")
    
    stats_file = output_path / "image_type_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"✓ 统计摘要已保存: {stats_file}")
    
    md_file = output_path / "image_type_report.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 图片类型分布统计报告\n\n")
        f.write(f"**总样本数**: {statistics['total_samples']}\n\n")
        f.write("## 类型分布\n\n")
        f.write("| 类型 | 数量 | 比例 |\n")
        f.write("|------|------|------|\n")
        
        sorted_types = sorted(
            statistics['type_distribution'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for img_type, data in sorted_types:
            f.write(f"| {img_type} | {data['count']} | {data['percentage']}% |\n")
        
        f.write("\n## 类型说明\n\n")
        for img_type, _ in sorted_types:
            try:
                type_enum = ImageType(img_type)
                strategy = ImageTypeStrategy.get_strategy(type_enum)
                description = strategy.get("description", "")
                priority = strategy.get("priority", "")
                edit_allowed = strategy.get("edit_allowed", "")
                
                f.write(f"### {img_type}\n\n")
                f.write(f"- **描述**: {description}\n")
                f.write(f"- **搜索优先级**: {priority}\n")
                f.write(f"- **编辑允许度**: {edit_allowed}\n\n")
            except:
                pass
    
    print(f"✓ Markdown报告已保存: {md_file}")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="统计原始数据中各图片类型的分布")
    parser.add_argument(
        "--data_path",
        type=str,
        default="path_to_Vision-SR1-47K",
        help="数据集路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="要分析的样本数量"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="path_to_image_type_analysis",
        help="输出目录"
    )
    parser.add_argument(
        "--no-gpt4o",
        action="store_true",
        help="不使用GPT-4o进行分类（使用关键词匹配）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从检查点恢复（如果存在）"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("图片类型分布统计工具")
    print("="*80)
    print(f"数据路径: {args.data_path}")
    print(f"样本数量: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print(f"分类方法: {'GPT-4o' if not args.no_gpt4o else '关键词匹配'}")
    print("="*80)
    print()
    
    checkpoint_file = Path(args.output_dir) / "classification_checkpoint.json"
    classified_samples = []
    
    if args.resume and checkpoint_file.exists():
        print(f"📂 发现检查点文件，正在加载...")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            classified_samples = json.load(f)
        print(f"✓ 已从检查点恢复 {len(classified_samples)} 个样本\n")
    else:
        samples = load_dataset(args.data_path, args.num_samples)
        
        classified_samples = classify_samples(samples, use_gpt4o=not args.no_gpt4o)
        
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(classified_samples, f, ensure_ascii=False, indent=2)
        print(f"✓ 检查点已保存: {checkpoint_file}\n")
    
    statistics = generate_statistics(classified_samples)
    
    save_results(classified_samples, statistics, args.output_dir)
    
    print("="*80)
    print("✅ 分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
