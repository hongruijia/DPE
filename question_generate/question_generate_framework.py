import argparse
import json
import os
import sys
from typing import List, Dict, Any
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework import QuestionGenerationFramework
from config import STORAGE_PATH


def load_vqa_dataset(data_path: str, max_samples: int = None) -> Dataset:

    datasets = []
    parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    parquet_files.sort()
    
    for parquet_file in parquet_files:
        print(f"Loading {parquet_file}...")
        dataset = Dataset.from_parquet(os.path.join(data_path, parquet_file))
        datasets.append(dataset)
    
    if len(datasets) > 1:
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets(datasets)
    else:
        combined_dataset = datasets[0]
    
    if max_samples:
        combined_dataset = combined_dataset.select(range(min(max_samples, len(combined_dataset))))
    
    print(f"Total samples loaded: {len(combined_dataset)}")
    return combined_dataset


def extract_sample_info(sample: Dict[str, Any], image_dir: str = None) -> Dict[str, str]:

    original_question = sample.get("problem", "")
    original_answer = sample.get("answer", "")
    original_question_type = sample.get("problem_type", "")  
    if "images" in sample:
        import tempfile
        from PIL import Image
        
        img = sample["images"]
        if isinstance(img, Image.Image):
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, f"temp_image_{id(img)}.png")
            img.save(image_path)
        else:
            image_path = str(img)
    elif "image_path" in sample:
        image_path = sample["image_path"]
    elif "image_id" in sample and image_dir:
        image_id = sample["image_id"]
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
    else:
        image_path = ""
    
    return {
        "original_question": original_question,
        "original_answer": original_answer,
        "original_question_type": original_question_type,
        "image_path": image_path
    }


def main(args):
    print("="*80)
    print("基于框架的问题生成系统")
    print("="*80)
    print(f"数据集路径: {args.data_path}")
    print(f"处理样本数: {args.num_samples}")
    print(f"输出文件: {args.save_name}_{args.suffix}.json")
    print("="*80)
    
    print("\n初始化问题生成框架...")
    framework = QuestionGenerationFramework(verbose=args.verbose)
    
    print(f"\n加载VQA数据集...")
    try:
        vqa_dataset = load_vqa_dataset(args.data_path, max_samples=args.max_samples)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    print(f"\n准备样本...")
    samples = []
    for i, sample in enumerate(vqa_dataset):
        if i >= args.num_samples:
            break
        
        sample_info = extract_sample_info(sample, image_dir=args.image_dir)
        
        if not sample_info["image_path"] or not os.path.exists(sample_info["image_path"]):
            print(f"警告: 样本 {i} 的图片路径无效或不存在: {sample_info['image_path']}")
            if not args.skip_missing:
                continue
        
        samples.append(sample_info)
    
    print(f"准备了 {len(samples)} 个有效样本")
    
    if not samples:
        print("没有有效样本，退出")
        return
    
    print(f"\n开始批量生成问题...")
    results = framework.batch_generate(samples, max_samples=args.num_samples)
    
    output_file = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\n保存结果到 {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✓ 成功生成 {len(results)} 个问题并保存")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return
    
    print("\n" + "="*80)
    print("生成统计")
    print("="*80)
    print(f"总样本数: {len(samples)}")
    print(f"成功生成: {len(results)}")
    print(f"成功率: {len(results)/len(samples)*100:.2f}%")
    
    type_counts = {}
    for result in results:
        qtype = result.get("question_type", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    print("\n问题类型分布:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count} ({count/len(results)*100:.1f}%)")
    
    print("="*80)
    
    if results:
        print("\n示例问题:")
        print("-"*80)
        example = results[0]
        print(f"类型: {example.get('question_type', 'N/A')}")
        print(f"问题: {example.get('question', 'N/A')}")
        print(f"答案: {example.get('answer', 'N/A')}")
        print(f"图片: [base64编码，长度 {len(example.get('image', ''))} 字符]")
        print("-"*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于框架的问题生成系统")
    
    parser.add_argument("--data_path", type=str, 
                       default="path_to_Vision-SR1-47K",
                       help="VQA数据集路径")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="图片目录（如果图片单独存储）")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="要处理的样本数")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="从数据集加载的最大样本数")
    
    parser.add_argument("--save_name", type=str, default="framework_generated",
                       help="输出文件名前缀")
    parser.add_argument("--suffix", type=str, default="v1",
                       help="输出文件名后缀")
    
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="是否输出详细信息")
    parser.add_argument("--skip_missing", action="store_true",
                       help="跳过缺失图片的样本")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n用户中断，退出...")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()


