import json
import os
import time
from typing import Dict, Any, Set, Optional
from threading import Lock
import hashlib


class IncrementalStorage:
    
    def __init__(self, output_dir: str, session_name: str):

        self.output_dir = output_dir
        self.session_name = session_name
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.data_file = os.path.join(output_dir, f"{session_name}.jsonl")
        self.progress_file = os.path.join(output_dir, f"{session_name}.progress")
        self.lock_file = os.path.join(output_dir, f"{session_name}.lock")
        
        self.write_lock = Lock()
        
        self.processed_samples: Set[str] = set()
        self.total_count = 0
        self.success_count = 0
        self.start_time = time.time()
        
        self._load_progress()
        
        print(f"[IncrementalStorage] 初始化完成")
        print(f"  数据文件: {self.data_file}")
        print(f"  进度文件: {self.progress_file}")
        print(f"  已处理: {len(self.processed_samples)} 条")
    
    def _load_progress(self):
        if os.path.exists(self.data_file):
            print(f"[IncrementalStorage] 检测到已有数据文件，加载中...")
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            sample_id = data.get("_sample_id", "")
                            if sample_id:
                                self.processed_samples.add(sample_id)
                        except json.JSONDecodeError as e:
                            print(f"  警告: 第{line_no}行数据损坏，跳过: {e}")
                print(f"  ✓ 加载了 {len(self.processed_samples)} 条已完成数据")
            except Exception as e:
                print(f"  警告: 加载数据文件失败: {e}")
        
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.total_count = progress.get("total_count", 0)
                    self.success_count = progress.get("success_count", len(self.processed_samples))
                    print(f"  ✓ 恢复进度: {self.success_count}/{self.total_count}")
            except Exception as e:
                print(f"  警告: 加载进度文件失败: {e}")
    
    def _save_progress(self):
        try:
            progress = {
                "total_count": self.total_count,
                "success_count": self.success_count,
                "processed_samples": list(self.processed_samples),
                "last_update": time.time(),
                "elapsed_time": time.time() - self.start_time
            }
            
            temp_file = self.progress_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            
            os.replace(temp_file, self.progress_file)
        except Exception as e:
            print(f"[IncrementalStorage] 警告: 保存进度失败: {e}")
    
    def get_sample_id(self, sample: Dict[str, Any]) -> str:

        key_parts = [
            sample.get("original_question", ""),
            sample.get("original_answer", ""),
            sample.get("image_path", "")
        ]
        key_str = "|".join(key_parts)
        sample_id = hashlib.md5(key_str.encode()).hexdigest()[:16]
        return sample_id
    
    def is_processed(self, sample: Dict[str, Any]) -> bool:

        sample_id = self.get_sample_id(sample)
        return sample_id in self.processed_samples
    
    def save_result(self, sample: Dict[str, Any], result: Dict[str, Any]) -> bool:

        sample_id = self.get_sample_id(sample)

        with self.write_lock:
            try:
                result_with_meta = {
                    "_sample_id": sample_id,
                    "_timestamp": time.time(),
                    "_original_question": sample.get("original_question", ""),
                    "_original_answer": sample.get("original_answer", ""),
                    **result
                }
                
                with open(self.data_file, 'a', encoding='utf-8') as f:
                    json_line = json.dumps(result_with_meta, ensure_ascii=False)
                    f.write(json_line + '\n')
                    f.flush()  
                
                self.processed_samples.add(sample_id)
                self.success_count += 1
                
                if self.success_count % 10 == 0:
                    self._save_progress()
                
                return True
                
            except Exception as e:
                print(f"[IncrementalStorage] 保存失败: {e}")
                return False
    
    def mark_failed(self, sample: Dict[str, Any], error: str):

        sample_id = self.get_sample_id(sample)
        
        failed_file = os.path.join(self.output_dir, f"{self.session_name}.failed.jsonl")
        
        try:
            with open(failed_file, 'a', encoding='utf-8') as f:
                failed_info = {
                    "_sample_id": sample_id,
                    "_timestamp": time.time(),
                    "_error": error,
                    "original_question": sample.get("original_question", ""),
                    "original_answer": sample.get("original_answer", ""),
                    "image_path": sample.get("image_path", "")
                }
                json_line = json.dumps(failed_info, ensure_ascii=False)
                f.write(json_line + '\n')
        except Exception as e:
            print(f"[IncrementalStorage] 记录失败样本时出错: {e}")
    
    def finalize(self):
        self._save_progress()
        
        final_json_file = os.path.join(self.output_dir, f"{self.session_name}_final.json")
        
        try:
            results = []
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except:
                        pass
            
            with open(final_json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"[IncrementalStorage] ✓ 最终数据已保存到: {final_json_file}")
        except Exception as e:
            print(f"[IncrementalStorage] 保存最终JSON失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        
        return {
            "total": self.total_count,
            "success": self.success_count,
            "failed": self.total_count - self.success_count,
            "success_rate": (self.success_count / self.total_count * 100) if self.total_count > 0 else 0,
            "elapsed_time": elapsed,
            "avg_time_per_sample": (elapsed / self.success_count) if self.success_count > 0 else 0,
            "processed_samples": len(self.processed_samples)
        }
    
    def print_stats(self):
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("📊 生成统计")
        print("="*80)
        print(f"总样本数:     {stats['total']}")
        print(f"成功生成:     {stats['success']} ✓")
        print(f"失败:         {stats['failed']} ✗")
        print(f"成功率:       {stats['success_rate']:.1f}%")
        print(f"总耗时:       {stats['elapsed_time']:.1f} 秒")
        print(f"平均耗时:     {stats['avg_time_per_sample']:.1f} 秒/条")
        
        if stats['success'] > 0:
            remaining = stats['total'] - stats['success']
            eta = remaining * stats['avg_time_per_sample']
            print(f"预计剩余时间: {eta:.1f} 秒 ({eta/60:.1f} 分钟)")
        
        print("="*80)

