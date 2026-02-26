import os
import json
import base64
import time
import glob
from typing import Dict, List, Optional, Any
from PIL import Image
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from agents import (
    PlannerAgent,
    ImageSearchAgent,
    ImageAnalysisAndSelectionAgent,
    ImageEditAgent,
    ImageSelectorAgent,
    QuestionGeneratorAgent,
    ValidationAgent
)
from config import STORAGE_PATH

def get_tmp_image_dir():
    base_dir = "path_to_your_tmp-image"
    suffix = os.environ.get("TMP_IMAGE_SUFFIX", "")
    if suffix:
        return os.path.join(base_dir, suffix)
    return base_dir

TMP_IMAGE_DIR = get_tmp_image_dir()


class IncrementalSaver:
    
    def __init__(self, output_dir: str, save_name: str, suffix: str, resume: bool = True):

        self.output_dir = output_dir
        self.save_name = save_name
        self.suffix = suffix
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.checkpoint_file = os.path.join(
            output_dir,
            f"{save_name}_{suffix}_checkpoint.json"
        )
        
        if resume and os.path.exists(self.checkpoint_file):
            existing_file = self._find_existing_incremental_file()
            if existing_file:
                self.incremental_file = existing_file
                print(f"[断点续传] 使用现有增量文件: {os.path.basename(self.incremental_file)}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.incremental_file = os.path.join(
                    output_dir, 
                    f"{save_name}_{suffix}_incremental_{timestamp}.jsonl"
                )
                print(f"⚠️  未找到现有增量文件，创建新文件")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.incremental_file = os.path.join(
                output_dir, 
                f"{save_name}_{suffix}_incremental_{timestamp}.jsonl"
            )
        
        self.final_file = os.path.join(
            output_dir,
            f"{save_name}_{suffix}.json"
        )
        
        self.lock = threading.Lock()
        
        if resume and os.path.exists(self.checkpoint_file):
            checkpoint = self._load_checkpoint_data()
            self.success_count = checkpoint.get("success_count", 0)
            self.failed_count = checkpoint.get("failed_count", 0)
            print(f"[断点续传] 恢复计数器: 成功={self.success_count}, 失败={self.failed_count}")
        else:
            self.success_count = 0
            self.failed_count = 0
        
        print(f"✓ 增量保存器初始化:")
        print(f"  - 增量文件: {os.path.basename(self.incremental_file)}")
        print(f"  - 最终文件: {os.path.basename(self.final_file)}")
        print(f"  - 检查点文件: {os.path.basename(self.checkpoint_file)}")
    
    def _find_existing_incremental_file(self) -> Optional[str]:
        pattern = f"{self.save_name}_{self.suffix}_incremental_*.jsonl"
        import glob
        matching_files = glob.glob(os.path.join(self.output_dir, pattern))
        
        if matching_files:
            latest_file = max(matching_files, key=os.path.getmtime)
            return latest_file
        return None
    
    def _load_checkpoint_data(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def save_one(self, result: Dict[str, Any], sample_index: int):

        with self.lock:
            result["_meta"] = {
                "sample_index": sample_index,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            with open(self.incremental_file, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
            
            self.success_count += 1
            print(f"[增量保存] ✓ 样本 {sample_index} 已保存 (累计成功: {self.success_count})")
    
    def save_failed(self, sample_index: int, error_msg: str):

        with self.lock:
            failed_record = {
                "_meta": {
                    "sample_index": sample_index,
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": error_msg
                }
            }
            
            with open(self.incremental_file, "a", encoding="utf-8") as f:
                json.dump(failed_record, f, ensure_ascii=False)
                f.write("\n")
            
            self.failed_count += 1
            print(f"[增量保存] ✗ 样本 {sample_index} 失败记录已保存 (累计失败: {self.failed_count})")
    
    def save_checkpoint(self, processed_indices: List[int], quota_state: Optional[Dict] = None):

        with self.lock:
            checkpoint = {
                "processed_indices": processed_indices,
                "success_count": self.success_count,
                "failed_count": self.failed_count,
                "last_update": datetime.now().isoformat()
            }
            
            if quota_state:
                checkpoint["quota_state"] = quota_state
            
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self) -> tuple[List[int], Optional[Dict]]:

        quota_state = None
        processed_indices_set = set()
        
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                print(f"[断点续传] 找到检查点文件:")
                print(f"  - 检查点记录: {len(checkpoint['processed_indices'])} 个样本")
                print(f"  - 成功: {checkpoint['success_count']}, 失败: {checkpoint['failed_count']}")
                
                quota_state = checkpoint.get("quota_state")
                if quota_state:
                    print(f"  - 配额状态已保存")
                
                processed_indices_set.update(checkpoint["processed_indices"])
        
        if os.path.exists(self.incremental_file):
            print(f"[断点续传] 从增量文件重建已处理索引...")
            with open(self.incremental_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            sample_idx = record.get('_meta', {}).get('sample_index')
                            if sample_idx is not None:
                                processed_indices_set.add(sample_idx)
                        except json.JSONDecodeError:
                            continue
            print(f"  - 增量文件记录: {len(processed_indices_set)} 个已处理样本（含成功和失败）")
        
        processed_indices = sorted(list(processed_indices_set))
        return processed_indices, quota_state
    
    def clean_tmp_images(self, auto_clean: bool = True) -> Dict[str, Any]:

        if not auto_clean:
            print("[临时图片] 跳过清理（auto_clean=False）")
            return {"cleaned": False, "reason": "auto_clean disabled"}
        
        tmp_dir = get_tmp_image_dir() 
        
        if not os.path.exists(tmp_dir):
            print("[临时图片] 目录不存在，无需清理")
            return {"cleaned": False, "reason": "directory not exists"}
        
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files_to_clean = []
        for pattern in image_patterns:
            files_to_clean.extend(glob.glob(os.path.join(tmp_dir, pattern)))
        
        file_count = len(files_to_clean)
        
        if file_count == 0:
            print("[临时图片] 目录为空，无需清理")
            return {"cleaned": False, "reason": "no files"}
        
        total_size = sum(os.path.getsize(f) for f in files_to_clean if os.path.exists(f))
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"\n[临时图片] 准备清理:")
        print(f"  - 目录: {os.path.basename(tmp_dir)}")
        print(f"  - 文件数量: {file_count}")
        print(f"  - 占用空间: {total_size_mb:.1f} MB")
        
        cleaned_count = 0
        for file_path in files_to_clean:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned_count += 1
            except Exception as e:
                print(f"[警告] 删除文件失败 {file_path}: {e}")
        
        print(f"[临时图片] ✓ 清理完成，删除 {cleaned_count}/{file_count} 个文件，释放 {total_size_mb:.1f} MB")
        
        return {
            "cleaned": True,
            "file_count": file_count,
            "cleaned_count": cleaned_count,
            "size_mb": total_size_mb
        }
    
    def finalize(self, auto_clean_tmp: bool = True) -> str:

        print(f"\n[最终合并] 合并增量结果...")
        
        results = []
        if os.path.exists(self.incremental_file):
            with open(self.incremental_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            if record.get("_meta", {}).get("success", False):
                                results.append(record)
                        except json.JSONDecodeError as e:
                            print(f"[警告] 解析行失败: {e}")
        
        with open(self.final_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[最终合并] ✓ 已保存 {len(results)} 条成功记录到: {self.final_file}")
        
        self.clean_tmp_images(auto_clean=auto_clean_tmp)
        
        return self.final_file


class OptimizedQuestionGenerationFramework:
    
    def __init__(
        self,
        verbose: bool = True,
        max_workers: int = 3,
        weakness_context: str = None,
        category_quotas: str = None,
        quota_manager = None,
        divergent_mode: bool = False
    ):

        self.verbose = verbose
        self.max_workers = max_workers
        self.weakness_context = self._load_weakness_context(weakness_context)
        self.quota_manager = quota_manager
        self.divergent_mode = divergent_mode
        
        self.quota_lock = threading.Lock()
        
        if self.verbose:
            if self.weakness_context:
                print(f"   ✓ 已加载弱点提示（{len(self.weakness_context)} 字符）")
            if self.quota_manager:
                print(f"   ✓ 已启用配额管理")
            if self.divergent_mode:
                print(f"   ✓ 已启用发散模式（从已有问题发散生成）")
            print(f"✓ 优化框架初始化完成 (并行度: {max_workers})")
    
    def _load_weakness_context(self, weakness_context: str) -> str:
        if not weakness_context:
            return ""
        
        if os.path.exists(weakness_context):
            try:
                with open(weakness_context, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if self.verbose:
                        print(f"   ✓ 从文件加载弱点提示: {weakness_context}")
                    return content
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  加载弱点提示文件失败: {e}")
                return ""
        else:
            return weakness_context
    
    def _create_worker_agents(self):
        return {
            'planner': PlannerAgent(weakness_context=self.weakness_context),
            'searcher': ImageSearchAgent(),
            'analysis_agent': ImageAnalysisAndSelectionAgent(),
            'editor': ImageEditAgent(),
            'selector': ImageSelectorAgent(),
            'generator': QuestionGeneratorAgent(weakness_context=self.weakness_context),
            'validator': ValidationAgent()
        }
    
    def log(self, message: str, level: str = "INFO"):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}][{level}] {message}")
    
    def image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def generate_question_from_sample(
        self,
        original_question: str,
        original_answer: str,
        original_image_path: str,
        original_question_type: str = "",
        use_original_image: bool = True,
        max_retries: int = 2,
        agents: Optional[Dict] = None,
        is_divergent: bool = False,
        image_type: str = "",
        divergent_inspiration: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:

        if agents is None:
            agents = self._create_worker_agents()
        
        if is_divergent or self.divergent_mode:
            self.log("🔄 发散生成模式：以源问题为参考，走完整流程", "INFO")
            if divergent_inspiration:
                self.log(f"   灵感来源: {divergent_inspiration.get('source_question', '')[:50]}...", "INFO")
                self.log(f"   目标类型: {image_type}", "INFO")
            
            from framework import QuestionGenerationFramework
            
            temp_framework = QuestionGenerationFramework(
                verbose=False,
                weakness_context=self.weakness_context
            )
            
            has_original_image = original_image_path and original_image_path.strip()
            
            if has_original_image:
                self.log(f"   ✓ 使用源问题的原图（保持类型: {image_type}）", "INFO")
            else:
                self.log(f"   ⚠ 无原图，将搜索/生成新图（类型可能变化）", "WARNING")
            
            result = temp_framework.generate_question_from_sample(
                original_question=original_question,
                original_answer=original_answer,
                original_image_path=original_image_path,  # ✅ 使用传入的原图
                original_question_type=original_question_type,
                use_original_image=use_original_image,    # ✅ 尊重调用者的选择
                max_retries=max_retries,
                preset_image_type=image_type  # 🎯 传递预设类型，保持类型一致
            )
            
            if result:

                result['generation_mode'] = 'divergent'
                if 'image_type' not in result or not result['image_type']:
                    result['image_type'] = image_type
                if divergent_inspiration:
                    result['divergent_inspiration'] = divergent_inspiration
                
            return result
        
        from framework import QuestionGenerationFramework
        
        temp_framework = QuestionGenerationFramework(
            verbose=False,
            weakness_context=self.weakness_context
        )
        
        return temp_framework.generate_question_from_sample(
            original_question=original_question,
            original_answer=original_answer,
            original_image_path=original_image_path,
            original_question_type=original_question_type,
            use_original_image=use_original_image,
            max_retries=max_retries
        )
    
    def batch_generate_with_incremental_save(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
        save_name: str = "generated",
        suffix: str = "v1",
        max_samples: Optional[int] = None,
        target_count: Optional[int] = None,
        resume: bool = True
    ) -> List[Dict[str, Any]]:

        saver = IncrementalSaver(output_dir, save_name, suffix, resume=resume)
        
        if resume:
            processed_indices, quota_state = saver.load_checkpoint()
            if self.quota_manager and quota_state:
                self.quota_manager.restore_state(quota_state)
        else:
            processed_indices = []
        
        success_count = 0
        if resume and target_count:
            import glob
            incremental_files = glob.glob(f"{output_dir}/{save_name}_{suffix}_incremental_*.jsonl")
            for file in incremental_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    record = json.loads(line)
                                    # 只统计成功的记录
                                    if record.get('_meta', {}).get('success', False):
                                        success_count += 1
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    self.log(f"读取增量文件 {file} 失败: {e}", "WARNING")
            self.log(f"📊 已有成功结果: {success_count}", "INFO")
        
        if target_count:
            total = len(samples)
            mode_desc = f"生成 {target_count} 个成功结果（已有 {success_count}）"
        else:
            total = len(samples) if max_samples is None else min(max_samples, len(samples))
            mode_desc = f"处理前 {total} 个输入样本"
        
        self.log(f"🎯 模式: {mode_desc}", "INFO")
        
        results = []
        for i, sample in enumerate(samples[:total]):
            if i in processed_indices:
                self.log(f"⏭️  样本 {i+1}/{total} 已处理，跳过", "INFO")
                continue
            
            self.log(f"\n{'='*80}")
            
            if target_count:
                self.log(f"处理样本 {i+1}/{total} (已成功: {success_count}/{target_count})")
            else:
                self.log(f"处理样本 {i+1}/{total}")
            
            if target_count and success_count >= target_count:
                self.log(f"✅ 已达到目标数量 {target_count}，停止生成", "INFO")
                break
            
            if self.quota_manager and self.quota_manager.is_complete():
                self.log(f"✋ 所有配额已完成，停止生成", "INFO")
                break
            
            if self.quota_manager:
                preset_image_type = sample.get('image_type', '')
                
                if preset_image_type:
                    if not self.quota_manager.can_generate_for_type(preset_image_type):
                        self.log(f"⚠️  类型 {preset_image_type} 配额已满，跳过样本 {i+1} (预检查)", "WARNING")
                        saver.save_failed(i, f"Quota full for type: {preset_image_type} (early check)")
                        processed_indices.append(i)
                        if (i + 1) % 5 == 0:
                            quota_state = self.quota_manager.get_state() if self.quota_manager else None
                            saver.save_checkpoint(processed_indices, quota_state)
                        continue
                    self.log(f"✓ 预检查通过: {preset_image_type} 配额未满，将调用 Planner (预设类型模式)", "INFO")
                
                from agents import PlannerAgent
                planner = PlannerAgent(weakness_context=self.weakness_context)
                
                image_path_for_planner = sample.get("image_path", "") or sample.get("original_image_path", "") or sample.get("image", "")
                
                plan = planner.create_plan(
                    original_question=sample.get("original_question", ""),
                    original_answer=sample.get("original_answer", ""),
                    image_path=image_path_for_planner,  # 🔧 兼容三种模式
                    original_question_type=sample.get("original_question_type", ""),
                    preset_image_type=preset_image_type  # 🎯 传入预设类型（如有则锁定，节省tokens）
                )
                
                if plan and 'image_type' in plan:
                    image_type = plan['image_type']
                    
                    if not self.quota_manager.can_generate_for_type(image_type):
                        self.log(f"⚠️  类型 {image_type} 配额已满，跳过样本 {i+1}", "WARNING")
                        saver.save_failed(i, f"Quota full for type: {image_type}")
                        processed_indices.append(i)
                        if (i + 1) % 5 == 0:
                            quota_state = self.quota_manager.get_state() if self.quota_manager else None
                            saver.save_checkpoint(processed_indices, quota_state)
                        continue
                    
                    self.log(f"📸 计划类型: {image_type} - 配额未满，开始生成", "INFO")
            
            try:
                is_divergent = sample.get("is_divergent", False) or self.divergent_mode
                image_type = sample.get("image_type", "")
                divergent_inspiration = sample.get("divergent_inspiration", None)
                
                result = self.generate_question_from_sample(
                    original_question=sample.get("original_question", ""),
                    original_answer=sample.get("original_answer", ""),
                    original_image_path=sample.get("image_path", "") or sample.get("image", ""),
                    original_question_type=sample.get("original_question_type", ""),
                    is_divergent=is_divergent,
                    image_type=image_type,
                    divergent_inspiration=divergent_inspiration
                )
                
                if result:
                    should_save = True
                    if self.quota_manager and 'image_type' in result:
                        image_type = result['image_type']
                        
                        if not self.quota_manager.can_generate_for_type(image_type):
                            self.log(f"⚠️  类型 {image_type} 配额已满，跳过此结果", "WARNING")
                            should_save = False
                        else:
                            self.quota_manager.increment_count(image_type)
                            self.log(f"📊 配额更新: {image_type}", "INFO")
                            if self.verbose:
                                progress = self.quota_manager.get_progress()
                                if progress['use_quotas'] and image_type in progress['by_type']:
                                    stats = progress['by_type'][image_type]
                                    self.log(f"   {image_type}: {stats['current']}/{stats['target']} ({stats['percentage']:.1f}%)", "INFO")
                    
                    if should_save:
                        saver.save_one(result, i)
                        results.append(result)
                        success_count += 1  
                        
                        if target_count:
                            self.log(f"✓ 样本 {i+1} 成功 ({success_count}/{target_count})", "SUCCESS")
                        else:
                            self.log(f"✓ 样本 {i+1} 成功", "SUCCESS")
                    else:
                        saver.save_failed(i, f"Quota exceeded for type: {result.get('image_type', 'unknown')}")
                else:
                    saver.save_failed(i, "Generation returned None")
                    self.log(f"✗ 样本 {i+1} 失败", "ERROR")
                
            except Exception as e:
                self.log(f"✗ 样本 {i+1} 异常: {e}", "ERROR")
                saver.save_failed(i, str(e))
            
            processed_indices.append(i)
            if (i + 1) % 5 == 0:
                quota_state = self.quota_manager.get_state() if self.quota_manager else None
                saver.save_checkpoint(processed_indices, quota_state)
            
            time.sleep(0.5) 
        
        quota_state = self.quota_manager.get_state() if self.quota_manager else None
        saver.save_checkpoint(processed_indices, quota_state)
        
        final_file = saver.finalize()
        
        self.log(f"\n{'='*80}")
        self.log(f"批量生成完成: 成功 {len(results)}/{total}")
        self.log(f"最终文件: {final_file}")
        
        return results
    
    def batch_generate_parallel(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
        save_name: str = "generated",
        suffix: str = "v1",
        max_samples: Optional[int] = None,
        target_count: Optional[int] = None,
        resume: bool = True
    ) -> List[Dict[str, Any]]:
        
        saver = IncrementalSaver(output_dir, save_name, suffix, resume=resume)
        
        if resume:
            processed_indices, quota_state = saver.load_checkpoint()
            if self.quota_manager and quota_state:
                self.quota_manager.restore_state(quota_state)
        else:
            processed_indices = []
        processed_set = set(processed_indices)
        
        success_count = 0
        if resume and target_count:
            import glob
            incremental_files = glob.glob(f"{output_dir}/{save_name}_{suffix}_incremental_*.jsonl")
            for file in incremental_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    record = json.loads(line)
                                    # 只统计成功的记录
                                    if record.get('_meta', {}).get('success', False):
                                        success_count += 1
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    self.log(f"读取增量文件 {file} 失败: {e}", "WARNING")
            self.log(f"📊 已有成功结果: {success_count}", "INFO")
        
        if target_count:
            total = len(samples)
            mode_desc = f"生成 {target_count} 个成功结果（已有 {success_count}）"
        else:
            total = len(samples) if max_samples is None else min(max_samples, len(samples))
            mode_desc = f"处理前 {total} 个输入样本"
        
        self.log(f"🎯 模式: {mode_desc}", "INFO")
        
        pending_samples = [
            (i, sample) for i, sample in enumerate(samples[:total])
            if i not in processed_set
        ]
        
        if not pending_samples:
            self.log("所有样本已处理完成！", "SUCCESS")
            final_file = saver.finalize(auto_clean_tmp=False)
            self.log(f"最终文件: {final_file}", "INFO")
            return []
        
        self.log(f"待处理样本: {len(pending_samples)}/{total}")
        self.log(f"并行度: {self.max_workers} 个线程")
        
        results = []
        completed_count = 0
        current_success_count = success_count  
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if target_count:
                batch_size = self.max_workers * 2
                pending_queue = list(pending_samples)
                all_futures = {}
                
                while pending_queue or all_futures:
                    if current_success_count >= target_count:
                        self.log(f"✅ 已达到目标数量 {target_count}，停止提交新任务", "INFO")
                        pending_queue = []
                    
                    if self.quota_manager and self.quota_manager.is_complete():
                        self.log(f"✋ 所有配额已完成，停止提交新任务", "INFO")
                        pending_queue = []
                    
                    while len(all_futures) < batch_size and pending_queue:
                        i, sample = pending_queue.pop(0)
                        future = executor.submit(self._process_single_sample, i, sample)
                        all_futures[future] = i
                    
                    if all_futures:
                        try:
                            done_futures = as_completed(all_futures, timeout=1)
                            for future in done_futures:
                                sample_index = all_futures.pop(future)
                                completed_count += 1
                                
                                try:

                                    result = future.result(timeout=1200)
                                    
                                    if result:
                                        should_save = True
                                        if self.quota_manager and 'image_type' in result:
                                            image_type = result['image_type']
                                            
                                            with self.quota_lock:  
                                                if not self.quota_manager.can_generate_for_type(image_type):
                                                    self.log(f"⚠️  类型 {image_type} 配额已满，跳过此结果", "WARNING")
                                                    should_save = False
                                                else:
                                                    self.quota_manager.increment_count(image_type)
                                                    if self.verbose:
                                                        progress = self.quota_manager.get_progress()
                                                        if progress['use_quotas'] and image_type in progress['by_type']:
                                                            stats = progress['by_type'][image_type]
                                                            self.log(f"   📊 {image_type}: {stats['current']}/{stats['target']} ({stats['percentage']:.1f}%)", "INFO")
                                        
                                        if should_save:
                                            saver.save_one(result, sample_index)
                                            results.append(result)
                                            current_success_count += 1
                                            self.log(f"✓ [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 成功 ({current_success_count}/{target_count})", "SUCCESS")
                                        else:
                                            saver.save_failed(sample_index, f"Quota exceeded for type: {result.get('image_type', 'unknown')}")
                                    else:
                                        saver.save_failed(sample_index, "Generation returned None")
                                        self.log(f"✗ [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 失败", "ERROR")
                                
                                except TimeoutError:
                                    self.log(f"⏱️  [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 超时（卡住>20分钟）", "ERROR")
                                    saver.save_failed(sample_index, "Task execution timeout (>20 minutes)")
                                
                                except Exception as e:
                                    self.log(f"✗ [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 异常: {e}", "ERROR")
                                    saver.save_failed(sample_index, str(e))
                                
                                processed_indices.append(sample_index)
                                if len(processed_indices) % 5 == 0:
                                    quota_state = self.quota_manager.get_state() if self.quota_manager else None
                                    saver.save_checkpoint(processed_indices, quota_state)
                                
                                break  
                        
                        except TimeoutError:
                            pass
            
            else:
                future_to_index = {
                    executor.submit(
                        self._process_single_sample,
                        i, sample
                    ): i
                    for i, sample in pending_samples
                }
                
                for future in as_completed(future_to_index):
                    sample_index = future_to_index[future]
                    completed_count += 1
                    
                    if self.quota_manager and self.quota_manager.is_complete():
                        self.log(f"✋ 所有配额已完成", "INFO")
                    
                    try:
                        result = future.result(timeout=1200)
                        
                        if result:
                            should_save = True
                            if self.quota_manager and 'image_type' in result:
                                image_type = result['image_type']
                                
                                with self.quota_lock:  
                                    if not self.quota_manager.can_generate_for_type(image_type):
                                        self.log(f"⚠️  类型 {image_type} 配额已满，跳过此结果", "WARNING")
                                        should_save = False
                                    else:
                                        self.quota_manager.increment_count(image_type)
                                        if self.verbose:
                                            progress = self.quota_manager.get_progress()
                                            if progress['use_quotas'] and image_type in progress['by_type']:
                                                stats = progress['by_type'][image_type]
                                                self.log(f"   📊 {image_type}: {stats['current']}/{stats['target']} ({stats['percentage']:.1f}%)", "INFO")
                            
                            if should_save:
                                saver.save_one(result, sample_index)
                                results.append(result)
                                self.log(f"✓ [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 成功", "SUCCESS")
                            else:
                                saver.save_failed(sample_index, f"Quota exceeded for type: {result.get('image_type', 'unknown')}")
                        else:
                            saver.save_failed(sample_index, "Generation returned None")
                            self.log(f"✗ [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 失败", "ERROR")
                    
                    except TimeoutError:
                        self.log(f"⏱️  [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 超时（卡住>20分钟）", "ERROR")
                        saver.save_failed(sample_index, "Task execution timeout (>20 minutes)")
                    
                    except Exception as e:
                        self.log(f"✗ [{completed_count}/{len(pending_samples)}] 样本 {sample_index+1} 异常: {e}", "ERROR")
                        saver.save_failed(sample_index, str(e))
                    
                    processed_indices.append(sample_index)
                    if completed_count % 5 == 0:
                        quota_state = self.quota_manager.get_state() if self.quota_manager else None
                        saver.save_checkpoint(processed_indices, quota_state)
        
        quota_state = self.quota_manager.get_state() if self.quota_manager else None
        saver.save_checkpoint(processed_indices, quota_state)
        
        final_file = saver.finalize()
        
        self.log(f"\n{'='*80}")
        self.log(f"并行生成完成: 成功 {len(results)}/{len(pending_samples)}")
        self.log(f"最终文件: {final_file}")
        
        return results
    
    def _process_single_sample(self, index: int, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        self.log(f"🔄 开始处理样本 {index+1}...", "INFO")
        
        try:
            is_divergent = sample.get("is_divergent", False) or self.divergent_mode
            image_type = sample.get("image_type", "")
            divergent_inspiration = sample.get("divergent_inspiration", None)
            
            if image_type and self.quota_manager:
                with self.quota_lock:  
                    if not self.quota_manager.can_generate_for_type(image_type):
                        self.log(f"⚠️  样本 {index+1}: 类型 {image_type} 配额已满，跳过执行（节省API）", "WARNING")
                        return None  
            
            result = self.generate_question_from_sample(
                original_question=sample.get("original_question", ""),
                original_answer=sample.get("original_answer", ""),
                original_image_path=sample.get("image_path", "") or sample.get("image", ""),
                original_question_type=sample.get("original_question_type", ""),
                agents=None,  
                is_divergent=is_divergent,
                image_type=image_type,
                divergent_inspiration=divergent_inspiration
            )
            
            return result
        
        except Exception as e:
            self.log(f"样本 {index+1} 处理异常: {e}", "ERROR")
            return None

