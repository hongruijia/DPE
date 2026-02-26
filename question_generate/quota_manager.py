import json
import os
from typing import Dict, Optional, Any
from collections import defaultdict


class QuotaManager:
    
    
    def __init__(
        self,
        total_samples: int,
        weights_file: Optional[str] = None,
        weights_dict: Optional[Dict[str, float]] = None
    ):
        self.total_samples = total_samples
        self.current_counts = defaultdict(int)
        self.target_quotas = {}
        

        weights = self._load_weights(weights_file, weights_dict)
        

        if weights:
            self._calculate_quotas(weights)
        else:

            self.use_quotas = False
            print("   ⚠️  未提供权重，将使用自然分布（不限制各类型数量）")
    
    def _load_weights(
        self,
        weights_file: Optional[str],
        weights_dict: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        
        if weights_dict:
            return weights_dict
        
        if weights_file and os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    content = f.read().strip()
                    

                    if content.startswith('{'):
                        return json.loads(content)
                    else:
                        print(f"   ⚠️  权重文件格式不正确: {weights_file}")
                        return None
            except Exception as e:
                print(f"   ⚠️  加载权重文件失败: {e}")
                return None
        
        return None
    
    def _calculate_quotas(self, weights: Dict[str, float]):
        
        self.use_quotas = True
        

        total_weight = sum(weights.values())
        if total_weight == 0:
            print("   ⚠️  权重总和为0，使用均匀分布")
            self.use_quotas = False
            return
        
        normalized_weights = {
            k: v / total_weight for k, v in weights.items()
        }
        

        self.weights = normalized_weights
        

        allocated_total = 0
        theoretical_quotas = {}  # 理论配额（包括为 0 的）
        
        for img_type, weight in normalized_weights.items():
            quota = int(self.total_samples * weight)
            theoretical_quotas[img_type] = quota
            
            if quota > 0:  # 只记录非零配额到 target_quotas
                self.target_quotas[img_type] = quota
                allocated_total += quota
        

        self.flexible_quota = self.total_samples - allocated_total
        self.flexible_used = 0
        

        print(f"\n   📊 目标配额分配情况（总样本数: {self.total_samples}）:")
        print(f"   {'='*60}")
        

        if not self.target_quotas:
            print(f"   ⚠️  警告: 样本数太少，所有类型的配额都被计算为 0")
            print(f"   ")
            print(f"   📋 权重配置如下（但配额均为 0）:")
            for img_type, weight in sorted(
                normalized_weights.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                theoretical = theoretical_quotas[img_type]
                print(f"      {img_type:20s}: 权重 {weight:6.1%}  →  配额 {theoretical} 条 ❌")
            

            min_weight = min(normalized_weights.values())
            suggested_min = int(1 / min_weight) + 1 if min_weight > 0 else 20
            print(f"   ")
            print(f"   💡 建议: 至少使用 {suggested_min} 个样本才能体现权重分布")
            print(f"   ")
        else:

            zero_count = len(theoretical_quotas) - len(self.target_quotas)
            
            for img_type, quota in sorted(
                self.target_quotas.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                weight = normalized_weights[img_type]
                print(f"      {img_type:20s}: 权重 {weight:6.1%}  →  配额 {quota:3d} 条 ✓")
            

            if zero_count > 0:
                print(f"   ")
                print(f"   ⚠️  以下 {zero_count} 个类型配额为 0（样本数不足）:")
                for img_type, quota in theoretical_quotas.items():
                    if quota == 0:
                        weight = normalized_weights[img_type]
                        print(f"      {img_type:20s}: 权重 {weight:6.1%}  →  配额 0 条 ❌")
        

        if self.flexible_quota > 0:
            print(f"   {'='*60}")
            print(f"   🔄 灵活配额: {self.flexible_quota:4d} 条 ({self.flexible_quota/self.total_samples*100:5.1f}%)")
            print(f"      用途: 用于不在上述权重中的其他图像类型")
        
        print(f"   {'='*60}")
    
    def can_generate_for_type(self, image_type: str) -> bool:
        
        if not self.use_quotas:
            return True
        
        if image_type not in self.target_quotas:

            if hasattr(self, 'flexible_quota'):
                return self.flexible_used < self.flexible_quota
            else:

                return True
        
        current = self.current_counts[image_type]
        target = self.target_quotas[image_type]
        
        return current < target
    
    def increment_count(self, image_type: str):
        
        self.current_counts[image_type] += 1
        

        if hasattr(self, 'flexible_quota') and image_type not in self.target_quotas:
            self.flexible_used += 1
    
    def get_progress(self) -> Dict[str, Dict]:
        
        progress = {}
        
        if not self.use_quotas:

            total_generated = sum(self.current_counts.values())
            return {
                'use_quotas': False,
                'total_generated': total_generated,
                'total_target': self.total_samples,
                'by_type': dict(self.current_counts)
            }
        
        for img_type, target in self.target_quotas.items():
            current = self.current_counts[img_type]
            progress[img_type] = {
                'current': current,
                'target': target,
                'percentage': (current / target * 100) if target > 0 else 0,
                'remaining': target - current
            }
        

        total_current = sum(self.current_counts.values())
        total_target = sum(self.target_quotas.values())
        
        result = {
            'use_quotas': True,
            'total_generated': total_current,
            'total_target': total_target,
            'by_type': progress
        }
        

        if hasattr(self, 'flexible_quota'):
            result['flexible_quota'] = {
                'total': self.flexible_quota,
                'used': self.flexible_used,
                'remaining': self.flexible_quota - self.flexible_used,
                'percentage': (self.flexible_used / self.flexible_quota * 100) if self.flexible_quota > 0 else 0
            }
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        
        state = {
            'use_quotas': self.use_quotas,
            'total_samples': self.total_samples,
            'current_counts': dict(self.current_counts),
            'target_quotas': dict(self.target_quotas) if hasattr(self, 'target_quotas') else {}
        }
        

        if hasattr(self, 'flexible_quota'):
            state['flexible_quota'] = self.flexible_quota
            state['flexible_used'] = self.flexible_used
        
        return state
    
    def restore_state(self, state: Dict[str, Any]):
        
        if not state:
            return
        
        print(f"\n[断点续传] 正在恢复配额状态...")
        

        saved_total = state.get('total_samples')
        saved_quotas = state.get('target_quotas', {})
        
        warnings = []
        

        if saved_total and saved_total != self.total_samples:
            warnings.append(
                f"⚠️  总样本数已改变: {saved_total} → {self.total_samples}"
            )
        

        if saved_quotas != self.target_quotas:

            all_types = set(saved_quotas.keys()) | set(self.target_quotas.keys())
            changed_types = []
            
            for img_type in all_types:
                old_quota = saved_quotas.get(img_type, 0)
                new_quota = self.target_quotas.get(img_type, 0)
                if old_quota != new_quota:
                    changed_types.append(f"{img_type}: {old_quota} → {new_quota}")
            
            if changed_types:
                warnings.append(
                    f"⚠️  配额已改变:\n      " + "\n      ".join(changed_types)
                )
        

        if warnings:
            print(f"\n{'='*80}")
            print(f"⚠️  警告: 检测到配额参数变化")
            print(f"{'='*80}")
            for warning in warnings:
                print(f"   {warning}")
            print(f"\n   💡 建议:")
            print(f"      - 如果是新的迭代轮次，建议使用新的输出目录")
            print(f"      - 如果要继续之前的任务，请使用相同的参数")
            print(f"      - 当前将使用新的配额设置，但保留已生成的计数")
            print(f"{'='*80}\n")
        

        if 'current_counts' in state:
            self.current_counts = defaultdict(int, state['current_counts'])
            print(f"   ✓ 已恢复各类型计数: {dict(self.current_counts)}")
        

        if 'flexible_used' in state and hasattr(self, 'flexible_quota'):
            self.flexible_used = state['flexible_used']
            print(f"   ✓ 已恢复灵活配额: {self.flexible_used}/{self.flexible_quota}")
        

        total_generated = sum(self.current_counts.values())
        print(f"\n   📊 当前状态:")
        print(f"      已生成总数: {total_generated}")
        print(f"      目标总数: {self.total_samples}")
        print(f"      剩余配额: {self.total_samples - total_generated}")
        print()
    
    def is_complete(self) -> bool:
        
        if not self.use_quotas:

            return sum(self.current_counts.values()) >= self.total_samples
        

        for img_type, target in self.target_quotas.items():
            if self.current_counts[img_type] < target:
                return False
        

        if hasattr(self, 'flexible_quota') and self.flexible_quota > 0:
            if self.flexible_used < self.flexible_quota:
                return False
        
        return True
    
    def get_status_message(self) -> str:
        
        progress = self.get_progress()
        
        if not progress['use_quotas']:
            return (f"已生成: {progress['total_generated']}/{progress['total_target']} "
                   f"({progress['total_generated']/progress['total_target']*100:.1f}%)")
        
        lines = [f"已生成: {progress['total_generated']}/{progress['total_target']}"]
        
        for img_type, stats in sorted(
            progress['by_type'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        ):
            lines.append(
                f"  {img_type:20s}: {stats['current']:3d}/{stats['target']:3d} "
                f"({stats['percentage']:5.1f}%)"
            )
        

        if 'flexible_quota' in progress:
            flex = progress['flexible_quota']
            lines.append(
                f"  {'[灵活配额]':20s}: {flex['used']:3d}/{flex['total']:3d} "
                f"({flex['percentage']:5.1f}%)"
            )
        
        return '\n'.join(lines)
    
    def get_distribution_summary(self) -> str:
        
        total = sum(self.current_counts.values())
        if total == 0:
            return "无数据"
        
        lines = [f"\n{'='*80}"]
        lines.append(f"📊 实际生成的图像类型分布")
        lines.append(f"{'='*80}")
        lines.append(f"总计: {total} 条")
        lines.append(f"{'-'*80}")
        

        sorted_types = sorted(
            self.current_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for img_type, count in sorted_types:
            percentage = count / total * 100
            

            in_weights = img_type in self.target_quotas if hasattr(self, 'target_quotas') else False
            marker = "✓" if in_weights else "🔄"
            

            if in_weights:
                target = self.target_quotas[img_type]
                status = "达标" if count >= target else f"未达标({count}/{target})"
                lines.append(
                    f"  {marker} {img_type:20s}: {count:3d} 条 ({percentage:5.1f}%)  [{status}]"
                )
            else:
                lines.append(
                    f"  {marker} {img_type:20s}: {count:3d} 条 ({percentage:5.1f}%)  [灵活配额]"
                )
        
        lines.append(f"{'-'*80}")
        

        if hasattr(self, 'weights') and self.weights:
            lines.append(f"\n📋 权重配置 vs 实际生成对比:")
            lines.append(f"{'-'*80}")
            
            for img_type in sorted(self.weights.keys(), key=lambda x: self.weights[x], reverse=True):
                weight = self.weights[img_type]
                expected_count = int(total * weight)
                actual_count = self.current_counts.get(img_type, 0)
                diff = actual_count - expected_count
                
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                lines.append(
                    f"  {img_type:20s}: 权重 {weight:6.1%}  "
                    f"期望 {expected_count:3d} → 实际 {actual_count:3d}  ({diff_str:>4s})"
                )
            
            lines.append(f"{'-'*80}")
        
        lines.append(f"{'='*80}")
        
        return '\n'.join(lines)
    
    def should_skip_sample(self, image_type: str) -> bool:
        
        return not self.can_generate_for_type(image_type)

