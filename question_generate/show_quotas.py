import json
import sys

def show_quotas(weights_file: str, total_samples: int):

    try:
        with open(weights_file, 'r') as f:
            weights = json.load(f)
    except Exception as e:
        print(f"❌ 加载权重文件失败: {e}")
        return
    

    total_weight = sum(weights.values())
    if total_weight == 0:
        print("❌ 权重总和为0")
        return
    
    print("=" * 60)
    print(f"📊 配额分配计算")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"权重文件: {weights_file}")
    print()
    

    quotas = {}
    for img_type, weight in weights.items():
        normalized = weight / total_weight
        quota = int(total_samples * normalized)
        quotas[img_type] = {
            'weight': weight,
            'normalized': normalized,
            'quota': quota,
            'percentage': normalized * 100
        }
    

    print("各类型配额分配:")
    print("-" * 60)
    print(f"{'类型':<25} {'权重':<10} {'配额':<10} {'占比':<10}")
    print("-" * 60)
    
    for img_type, info in sorted(quotas.items(), key=lambda x: x[1]['quota'], reverse=True):
        print(f"{img_type:<25} {info['weight']:<10.3f} {info['quota']:<10d} {info['percentage']:>6.1f}%")
    
    total_allocated = sum(q['quota'] for q in quotas.values())
    print("-" * 60)
    print(f"{'总计':<25} {total_weight:<10.3f} {total_allocated:<10d} {100.0:>6.1f}%")
    print("=" * 60)
    

    remainder = total_samples - total_allocated
    if remainder > 0:
        print(f"\n⚠️  注意: 由于整数舍入，有 {remainder} 条未分配（将自然生成）")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python show_quotas.py <权重文件路径> <总样本数>")
        print()
        print("示例:")
        print("  python show_quotas.py generation_weights.json 100")
        print("  python show_quotas.py ../question_evaluate/weakness_analysis_iter1/generation_weights.json 1000")
        sys.exit(1)
    
    weights_file = sys.argv[1]
    total_samples = int(sys.argv[2])
    
    show_quotas(weights_file, total_samples)


