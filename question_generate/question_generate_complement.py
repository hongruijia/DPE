import argparse
import json
import os
import sys
from typing import List, Dict, Any
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework_optimized import OptimizedQuestionGenerationFramework
from config import STORAGE_PATH
from quota_manager import QuotaManager


def load_generated_questions(checkpoint_file: str) -> Dict[str, Any]:
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    return checkpoint


def load_final_results(final_file: str) -> List[Dict[str, Any]]:
    with open(final_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def analyze_quota_gaps(checkpoint_file: str, quota_file: str, total_samples: int) -> Dict[str, int]:
    checkpoint = load_generated_questions(checkpoint_file)
    quota_state = checkpoint.get('quota_state', {})
    
    current_counts = quota_state.get('current_counts', {})
    target_quotas = quota_state.get('target_quotas', {})
    
    print("\n" + "="*80)
    print("📊 配额缺口分析")
    print("="*80)
    
    gaps = {}
    for category, target in target_quotas.items():
        current = current_counts.get(category, 0)
        gap = target - current
        if gap > 0:
            gaps[category] = gap
            print(f"  {category:20s}: 当前 {current:3d}/{target:3d} | 缺口: {gap:3d}")
        else:
            print(f"  {category:20s}: 当前 {current:3d}/{target:3d} | ✓ 已达标")
    
    if gaps:
        print(f"\n总缺口: {sum(gaps.values())} 个问题")
    else:
        print("\n✓ 所有类型都已达到配额")
    
    print("="*80)
    return gaps


def group_by_image_type(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}
    for result in results:
        image_type = result.get('image_type', 'unknown')
        if image_type not in grouped:
            grouped[image_type] = []
        grouped[image_type].append(result)
    return grouped


def create_divergent_samples(
    source_questions: List[Dict[str, Any]], 
    num_samples: int,
    image_type: str
) -> List[Dict[str, Any]]:

    print(f"\n为 {image_type} 创建 {num_samples} 个发散样本（有源问题）...")
    print(f"  可用源问题数: {len(source_questions)}")
    
    samples = []
    num_sources = len(source_questions)
    
    if num_sources == 0:
        return samples
    
    samples_per_source = num_samples // num_sources
    remaining = num_samples % num_sources
    
    print(f"  策略: 每个源问题生成 {samples_per_source} 个变体")
    if remaining > 0:
        print(f"        前 {remaining} 个源问题额外生成 1 个")
    print(f"  生成流程: plan→搜图/生图→分析→编辑→生成→验证(ValidationAgent)")
    
    for source_idx, source in enumerate(source_questions):
        variants_count = samples_per_source + (1 if source_idx < remaining else 0)
        
        for variant_idx in range(variants_count):

            sample = {
                "original_question": source.get('question', ''),  
                "original_answer": source.get('answer', ''),  
                "original_question_type": source.get('question_type', ''),
                "image_type": image_type,  
                "is_divergent": True,  
                    "source_id": source.get('id', source_idx),  
                    "variant_index": variant_idx,  
                "divergent_inspiration": { 
                    "source_question": source.get('question', ''),
                    "source_answer": source.get('answer', ''),
                    "source_image_path": source.get('image_path', ''),
                        "variant_number": variant_idx + 1,
                        "total_variants": variants_count
                    }
                }
            samples.append(sample)
    
    return samples


def create_pure_divergent_samples(
    num_samples: int,
    image_type: str
) -> List[Dict[str, Any]]:

    print(f"\n为 {image_type} 创建 {num_samples} 个纯发散样本（无源问题）...")
    print(f"  策略: 完全自由生成，根据图像类型特点创造新问题")
    print(f"  生成流程: plan→搜图/生图→分析→编辑→生成→验证(ValidationAgent)")
    print(f"  质量保证: ①Planner规划 ②配额检查(2次) ③ValidationAgent验证")
    
    samples = []
    
    for i in range(num_samples):

        sample = {
            "image_type": image_type, 
            "is_pure_divergent": True,  
            "is_divergent": True,  
            "sample_index": i,
            "original_question": "",
            "original_answer": "",
            "original_image_path": "",
            "original_question_type": ""
        }
        samples.append(sample)
    
    return samples


def main(args):
    print("="*80)
    print("🔄 补充问题生成系统")
    print("="*80)
    print(f"检查点文件: {args.checkpoint_file}")
    print(f"最终结果文件: {args.final_file}")
    print(f"配额文件: {args.quota_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"并行模式: {'✓ 启用' if args.parallel else '✗ 串行'}")
    if args.parallel:
        print(f"并行度: {args.max_workers} 个线程")
    print("="*80)
    
    with open(args.quota_file, 'r', encoding='utf-8') as f:
        quota_data = json.load(f)
    
    if isinstance(quota_data, dict):
        if 'quotas' in quota_data:
            total_samples = sum(quota_data['quotas'].values())
        else:
            total_samples = sum(quota_data.values())
    else:
        print("❌ 配额文件格式错误")
        return
    
    print(f"\n从配额文件推断总目标样本数: {total_samples}")
    
    gaps = analyze_quota_gaps(args.checkpoint_file, args.quota_file, total_samples)
    
    if not gaps:
        print("\n✅ 所有类型都已达到配额，无需补充生成")
        return
    
    print("\n加载已生成的问题...")
    final_results = load_final_results(args.final_file)
    print(f"已加载 {len(final_results)} 个问题")
    
    print("\n按图像类型分组...")
    grouped_questions = group_by_image_type(final_results)
    for img_type, questions in grouped_questions.items():
        print(f"  {img_type:20s}: {len(questions):3d} 个问题")
    
    print("\n" + "="*80)
    print("📝 真正的动态遍历策略")
    print("="*80)
    print("策略: 遍历所有源问题 → 检查image_type配额 → 决定生成数量")
    print("优点: ① 不浪费tokens ② 覆盖所有源问题 ③ 配额满自动跳过")
    print("="*80)
    
    all_samples = []
    
    samples_with_source = 0
    samples_without_source = 0
    type_processed_count = {category: 0 for category in gaps.keys()}
    type_skipped_count = {category: 0 for category in gaps.keys()}
    
    print(f"\n📊 配额缺口:")
    for category, gap in sorted(gaps.items(), key=lambda x: -x[1]):
        num_sources = len(grouped_questions.get(category, []))
        if num_sources > 0:
            print(f"  {category:20s}: 需要 {gap:3d} 个, 有 {num_sources:3d} 个源问题")
        else:
            print(f"  {category:20s}: 需要 {gap:3d} 个, 无源问题（纯发散）")
    
    print(f"\n开始遍历所有源问题...")
    
    all_types_sources = {}
    for img_type, questions in grouped_questions.items():
        all_types_sources[img_type] = len(questions)
    
    total_sources = sum(all_types_sources.values())
    print(f"\n所有类型的源问题分布:")
    for img_type, count in sorted(all_types_sources.items(), key=lambda x: -x[1]):
        in_gaps = "✅ 需要" if img_type in gaps else "❌ 配额已满/不需要"
        print(f"  {img_type:20s}: {count:3d} 个源问题  {in_gaps}")
    
    print(f"\n总共有 {total_sources} 个源问题")
    print(f"策略: 只处理需要的类型（gaps中的），其他类型跳过")
    
    for category in gaps.keys():
        if category not in grouped_questions or len(grouped_questions[category]) == 0:
            continue
        
        sources = grouped_questions[category]
        gap = gaps[category]
        num_sources = len(sources)
        
        print(f"\n📌 遍历 {category} 类型的 {num_sources} 个源问题 (需要补充 {gap} 个)")
        
        if gap >= num_sources:
            variants_per_source = gap // num_sources
            extra_variants = gap % num_sources
            sources_to_use = num_sources
            print(f"   策略: 使用所有 {sources_to_use} 个源问题")
            print(f"   基础分配: 每个源问题生成 {variants_per_source} 个变体")
            if extra_variants > 0:
                print(f"   额外分配: 前 {extra_variants} 个源问题额外生成 1 个")
        else:
            variants_per_source = 1
            extra_variants = 0
            sources_to_use = gap
            print(f"   策略: 只使用前 {sources_to_use} 个源问题（源问题充足）")
            print(f"   基础分配: 每个源问题生成 1 个变体")
        
        for source_idx, source in enumerate(sources):
            if source_idx >= sources_to_use:
                break
            
            variants_count = variants_per_source
            if source_idx < extra_variants:
                variants_count += 1
            
            for variant_idx in range(variants_count):
                original_image = source.get('image_path', '') or source.get('image', '')
                
                sample = {
                    "original_question": source.get('question', ''),
                    "original_answer": source.get('answer', ''),
                    "original_image_path": original_image,  
                    "original_question_type": source.get('question_type', ''),
                    "image_type": category,  
                    "is_divergent": True,
                    "use_original_image": True, 
                    "source_id": source.get('id', source_idx),
                    "variant_index": variant_idx,
                    "divergent_inspiration": {
                        "source_question": source.get('question', ''),
                        "source_answer": source.get('answer', ''),
                        "source_image_path": original_image,
                        "variant_number": variant_idx + 1,
                        "total_variants": variants_count
                    }
                }
                all_samples.append(sample)
                samples_with_source += 1
                type_processed_count[category] += 1
        
        print(f"   ✓ 处理完成，创建了 {type_processed_count[category]} 个样本")
    
    for category, gap in gaps.items():
        if category in grouped_questions and len(grouped_questions[category]) > 0:
            continue
        
        print(f"\n📌 {category} 类型无源问题，创建 {gap} 个纯发散样本")
        samples = create_pure_divergent_samples(gap, category)
        all_samples.extend(samples)
        samples_without_source += len(samples)
        type_processed_count[category] = gap
    
    if not all_samples:
        print("\n❌ 没有可用的发散样本")
        return
    
    print(f"\n{'='*80}")
    print(f"✅ 所有源问题遍历完成")
    print(f"{'='*80}")
    print(f"总计创建: {len(all_samples)} 个样本")
    print(f"  - 从源问题发散: {samples_with_source} 个")
    print(f"  - 纯发散生成: {samples_without_source} 个")
    print(f"\n各类型样本分布:")
    for category, count in sorted(type_processed_count.items(), key=lambda x: -x[1]):
        print(f"  {category:20s}: {count:3d} 个样本")
    
    print(f"\n💡 生成时框架会:")
    print(f"  ① 预检查: 每个样本生成前检查image_type配额（Layer 0）")
    print(f"  ② 配额满跳过: 某类型配额满后，该类型样本自动跳过")
    print(f"  ③ 继续处理: 跳过不影响其他类型的处理")
    print(f"  ④ 自动停止: 所有配额满或样本处理完，自动停止")
    
    weakness_context = None
    if args.weakness_context:
        if os.path.exists(args.weakness_context):
            with open(args.weakness_context, 'r', encoding='utf-8') as f:
                weakness_context = f.read()
            print(f"\n✓ 加载弱点上下文文件: {args.weakness_context}")
        else:
            weakness_context = args.weakness_context
            print(f"\n✓ 使用弱点上下文内容（{len(weakness_context)} 字符）")
    
    print("\n初始化生成框架...")

    gap_total = sum(gaps.values())
    gap_weights = {img_type: gap / gap_total for img_type, gap in gaps.items()}
    
    print(f"\n📋 补充生成配额（仅包含需要补充的类型）:")
    for img_type, gap in sorted(gaps.items(), key=lambda x: -x[1]):
        weight = gap_weights[img_type]
        print(f"  {img_type:20s}: {gap:3d} 个 (权重: {weight:.1%})")
    
    quota_manager = QuotaManager(
        total_samples=gap_total,
        weights_dict=gap_weights  
    )
    
    quota_manager.use_quotas = True
    quota_manager.flexible_quota = 0  
    quota_manager.flexible_used = 0
    
    framework = OptimizedQuestionGenerationFramework(
        verbose=args.verbose,
        max_workers=args.max_workers,
        weakness_context=weakness_context,
        category_quotas=args.quota_file,
        quota_manager=quota_manager,
        divergent_mode=True 
    )
    
    checkpoint = load_generated_questions(args.checkpoint_file)
    base_suffix = checkpoint.get('suffix', 'complement')
    complement_suffix = f"{base_suffix}-complement"
    
    print(f"\n{'='*80}")
    print(f"📊 生成计划")
    print(f"{'='*80}")
    
    actual_quota_needed = sum(gaps.values())
    
    print(f"  目标: 补充 {actual_quota_needed} 个问题")
    print(f"  样本: {len(all_samples)} 个（精确创建）")
    print(f"  策略: 动态生成 + 配额检查")
    print(f"  停止条件:")
    print(f"    ① 所有样本处理完")
    print(f"    ② 所有配额已满（框架自动停止）")
    print(f"\n  工作流程:")
    print(f"    样本1 → 检查配额 → 生成 → 保存 → 更新配额")
    print(f"    样本2 → 检查配额 → 生成 → 保存 → 更新配额")
    print(f"    ...")
    print(f"    样本N → 检查配额 → 已满跳过 ✋")
    print(f"{'='*80}")
    
    if args.parallel:
        print("\n🚀 开始并行生成补充问题...")
        results = framework.batch_generate_parallel(
            samples=all_samples,
            output_dir=args.output_dir,
            save_name=args.save_name,
            suffix=complement_suffix,
            max_samples=None, 
            target_count=None, 
            resume=not args.no_resume
        )
    else:
        print("\n📝 开始串行生成补充问题...")
        results = framework.batch_generate_with_incremental_save(
            samples=all_samples,
            output_dir=args.output_dir,
            save_name=args.save_name,
            suffix=complement_suffix,
            max_samples=None,  
            target_count=None,  
            resume=not args.no_resume
        )
    
    print("\n" + "="*80)
    print("📊 补充生成统计")
    print("="*80)
    print(f"计划生成: {len(all_samples)}")
    print(f"实际生成: {len(results)}")
    if len(all_samples) > 0:
        print(f"成功率: {len(results)/len(all_samples)*100:.2f}%")
    
    if args.merge and len(results) > 0:
        print("\n" + "="*80)
        print("🔗 合并到原始结果")
        print("="*80)
        
        original_results = load_final_results(args.final_file)
        print(f"原始问题数: {len(original_results)}")
        
        complement_file = os.path.join(
            args.output_dir, 
            f"{args.save_name}_{complement_suffix}.json"
        )
        if os.path.exists(complement_file):
            with open(complement_file, 'r', encoding='utf-8') as f:
                complement_results = json.load(f)
            print(f"补充问题数: {len(complement_results)}")
            
            merged_results = original_results + complement_results
            
            merged_file = os.path.join(
                args.output_dir,
                f"{args.save_name}_{base_suffix}-merged.json"
            )
            with open(merged_file, 'w', encoding='utf-8') as f:
                json.dump(merged_results, f, ensure_ascii=False, indent=2)
            
            print(f"合并后总数: {len(merged_results)}")
            print(f"保存到: {merged_file}")
            
            type_counts = {}
            image_type_counts = {}
            for result in merged_results:
                qtype = result.get("question_type", "unknown")
                type_counts[qtype] = type_counts.get(qtype, 0) + 1
                
                img_type = result.get("image_type", "unknown")
                image_type_counts[img_type] = image_type_counts.get(img_type, 0) + 1
            
            print("\n问题类型分布:")
            for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {qtype}: {count} ({count/len(merged_results)*100:.1f}%)")
            
            print("\n图像类型分布:")
            for img_type, count in sorted(image_type_counts.items(), key=lambda x: x[1], reverse=True):
                target = quota_manager.target_quotas.get(img_type, 0) if hasattr(quota_manager, 'target_quotas') else 0
                if target > 0:
                    print(f"  {img_type:20s}: {count:3d}/{target:3d} ({count/target*100:.1f}%)")
                else:
                    print(f"  {img_type:20s}: {count:3d}")
    
    print("="*80)
    print("✅ 补充生成完成！")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="补充问题生成系统")
    
    parser.add_argument("--checkpoint-file", type=str, required=True,
                       help="已生成问题的检查点文件路径")
    parser.add_argument("--final-file", type=str, required=True,
                       help="已生成问题的最终结果文件路径")
    parser.add_argument("--quota-file", type=str, required=True,
                       help="配额文件路径")
    
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--save-name", type=str, default="optimized_generated",
                       help="输出文件名前缀")
    
    parser.add_argument("--weakness-context", type=str, default=None,
                       help="弱点提示文件路径")
    
    parser.add_argument("--parallel", action="store_true",
                       help="启用并行处理")
    parser.add_argument("--max-workers", type=int, default=5,
                       help="并行线程数")
    parser.add_argument("--no-resume", action="store_true",
                       help="禁用断点续传")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="详细输出")
    
    parser.add_argument("--merge", action="store_true", default=True,
                       help="自动合并到原始结果（默认启用）")
    parser.add_argument("--no-merge", dest='merge', action="store_false",
                       help="不合并到原始结果")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断！")
        print("💾 已生成的数据已安全保存")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

