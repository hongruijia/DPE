import argparse
import json
import os
import sys
from typing import List, Dict, Any
from datasets import Dataset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework_optimized import OptimizedQuestionGenerationFramework
from config import STORAGE_PATH
from quota_manager import QuotaManager


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
    print("🚀 优化的问题生成系统")
    print("="*80)
    print(f"数据集路径: {args.data_path}")
    print(f"处理样本数: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出文件名: {args.save_name}_{args.suffix}")
    print(f"并行模式: {'✓ 启用' if args.parallel else '✗ 串行'}")
    if args.parallel:
        print(f"并行度: {args.max_workers} 个线程")
    

    resume = not args.no_resume
    print(f"断点续传: {'✓ 启用' if resume else '✗ 禁用'}")
    

    weakness_context = args.weakness_prompt or args.weakness_context
    category_quotas = args.image_type_quotas or args.category_quotas
    
    if weakness_context:
        print(f"弱点提示: {weakness_context}")
    if category_quotas:
        print(f"图像类型配额: {category_quotas}")
    

    quota_manager = None
    if category_quotas:
        print(f"\n📊 初始化配额管理器...")
        quota_manager = QuotaManager(
            total_samples=args.num_samples,
            weights_file=category_quotas
        )
    
    print("="*80)
    

    print("\n初始化优化框架...")
    framework = OptimizedQuestionGenerationFramework(
        verbose=args.verbose,
        max_workers=args.max_workers,
        weakness_context=weakness_context,
        category_quotas=category_quotas,
        quota_manager=quota_manager
    )
    

    print(f"\n加载VQA数据集...")
    try:
        vqa_dataset = load_vqa_dataset(args.data_path, max_samples=args.max_samples)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    

    print(f"\n准备样本...")
    print(f"目标模式: {args.target_mode}")
    

    if args.target_mode == 'input':

        max_candidates = args.num_samples
        print(f"  - 模式: 处理前 {args.num_samples} 个输入样本")
    else:


        max_candidates = int(args.num_samples * 1.0) + 10
        print(f"  - 模式: 生成 {args.num_samples} 个成功结果")
        print(f"  - 准备 {max_candidates} 个候选样本（考虑失败率）")
    
    samples = []
    for i, sample in enumerate(vqa_dataset):
        if len(samples) >= max_candidates:
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
    

    print(f"\n{'='*80}")
    if args.parallel:
        print("🚀 开始并行生成问题（增量保存）...")
        results = framework.batch_generate_parallel(
            samples=samples,
            output_dir=args.output_dir,
            save_name=args.save_name,
            suffix=args.suffix,
            max_samples=args.num_samples if args.target_mode == 'input' else None,
            target_count=args.num_samples if args.target_mode == 'output' else None,
            resume=resume
        )
    else:
        print("📝 开始串行生成问题（增量保存）...")
        results = framework.batch_generate_with_incremental_save(
            samples=samples,
            output_dir=args.output_dir,
            save_name=args.save_name,
            suffix=args.suffix,
            max_samples=args.num_samples if args.target_mode == 'input' else None,
            target_count=args.num_samples if args.target_mode == 'output' else None,
            resume=resume
        )
    

    print("\n" + "="*80)
    print("📊 生成统计")
    print("="*80)
    print(f"总样本数: {len(samples)}")
    print(f"本次生成: {len(results)}")
    if len(samples) > 0:
        print(f"本次成功率: {len(results)/len(samples)*100:.2f}%")
    

    final_file = os.path.join(args.output_dir, f"{args.save_name}_{args.suffix}.json")
    if os.path.exists(final_file):
        with open(final_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        
        print(f"\n累计总成功: {len(all_results)} 条")
        

        type_counts = {}
        for result in all_results:
            qtype = result.get("question_type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        print("\n问题类型分布:")
        for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {qtype}: {count} ({count/len(all_results)*100:.1f}%)")
        

        if quota_manager:
            print(quota_manager.get_distribution_summary())
        

        if all_results:
            print("\n" + "="*80)
            print("📝 示例问题:")
            print("-"*80)
            example = all_results[0]
            print(f"类型: {example.get('question_type', 'N/A')}")
            print(f"问题: {example.get('question', 'N/A')}")
            print(f"答案: {example.get('answer', 'N/A')}")
            if 'image' in example:
                print(f"图片: [base64编码，长度 {len(example.get('image', ''))} 字符]")
            print("-"*80)
    
    print("="*80)
    print("✅ 全部完成！")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="优化的问题生成系统")
    

    parser.add_argument("--data_path", type=str, 
                       default="path_to_Vision-SR1-47K",
                       help="VQA数据集路径")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="图片目录（如果图片单独存储）")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="要处理的样本数")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="从数据集加载的最大样本数")
    

    parser.add_argument("--output_dir", type=str,
                       default=f"{STORAGE_PATH}/generated_question",
                       help="输出目录")
    parser.add_argument("--save_name", type=str, default="framework_generated",
                       help="输出文件名前缀")
    parser.add_argument("--suffix", type=str, default="v1",
                       help="输出文件名后缀")
    

    parser.add_argument("--parallel", action="store_true",
                       help="启用并行处理（多线程）")
    parser.add_argument("--max_workers", type=int, default=3,
                       help="并行处理的最大线程数（建议3-5）")
    parser.add_argument("--no-resume", action="store_true",
                       help="禁用断点续传（总是从头开始）")
    

    parser.add_argument("--verbose", action="store_true", default=True,
                       help="是否输出详细信息")
    parser.add_argument("--skip_missing", action="store_true",
                       help="跳过缺失图片的样本")
    parser.add_argument("--no-clean-tmp", action="store_true",
                       help="禁用自动清理临时图片（默认会清理）")
    

    parser.add_argument("--weakness_context", type=str, default=None,
                       help="弱点提示文件路径（用于针对性生成）")
    parser.add_argument("--weakness_prompt", type=str, default=None,
                       help="弱点提示文件路径（同 --weakness_context）")
    parser.add_argument("--category_quotas", type=str, default=None,
                       help="图像类型配额（JSON文件路径或JSON字符串）")
    parser.add_argument("--image_type_quotas", type=str, default=None,
                       help="图像类型配额（同 --category_quotas）")
    

    parser.add_argument("--target-mode", type=str, 
                       choices=['input', 'output'], 
                       default='input',
                       help="目标模式：'input'=处理N个输入样本（快速），'output'=生成N个成功结果（确保数量）")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断！")
        print("💾 已生成的数据已安全保存到增量文件")
        print("📌 下次运行时使用 --resume 参数可从断点继续")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

