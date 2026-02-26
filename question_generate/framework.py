import os
import base64
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
from io import BytesIO

from agents import (
    PlannerAgent,
    ImageSearchAgent,
    ImageAnalysisAndSelectionAgent,
    ImageEditAgent,
    ImageSelectorAgent,
    QuestionGeneratorAgent,
    ValidationAgent
)


class QuestionGenerationFramework:
    
    def __init__(self, verbose: bool = True, weakness_context: str = ""):

        self.verbose = verbose
        self.weakness_context = weakness_context
        
        # 初始化各个Agent，传递weakness_context
        self.planner = PlannerAgent(weakness_context=weakness_context)
        self.searcher = ImageSearchAgent()
        self.analysis_agent = ImageAnalysisAndSelectionAgent()  # 新的合并Agent
        self.editor = ImageEditAgent()
        self.selector = ImageSelectorAgent()
        self.generator = QuestionGeneratorAgent(weakness_context=weakness_context)
        self.validator = ValidationAgent()
        
        if self.verbose:
            if self.weakness_context:
                print(f"✓ 框架初始化完成（已加载弱点提示，{len(self.weakness_context)} 字符）")
            else:
                print("✓ 框架初始化完成")
    
    def log(self, message: str, level: str = "INFO"):
        if self.verbose:
            print(f"[{level}] {message}")
    
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
        preset_image_type: str = ""
    ) -> Optional[Dict[str, Any]]:

        self.log("="*80)
        self.log("开始生成新问题")
        self.log(f"原始问题: {original_question}")
        self.log(f"原始答案: {original_answer}")
        
        self.log("\n[步骤 1/7] 创建生成计划...")
        if original_question_type:
            self.log(f"原始问题类型: {original_question_type}")
        if preset_image_type:
            self.log(f"预设图像类型: {preset_image_type} (锁定)")
        
        plan = self.planner.create_plan(
            original_question=original_question,
            original_answer=original_answer,
            image_path=original_image_path,
            original_question_type=original_question_type,
            preset_image_type=preset_image_type  # 🎯 传递预设类型
        )
        
        if not plan:
            self.log("计划创建失败", "ERROR")
            return None
        
        plan_summary = plan.get('plan_summary', '生成基于图像的问题')
        question_type = plan.get('question_type', 'numerical')
        
        self.log(f"计划创建成功: {plan_summary}")
        self.log(f"问题类型: {question_type}")
        
        if not plan.get('search_related_query') and not plan.get('search_new_element_query'):
            self.log("警告: 计划中没有搜索查询，将使用默认策略", "WARNING")
        
        self.log("\n[步骤 2/7] 搜索相关图片...")
        related_query = plan.get("search_related_query", "")
        related_images = []
        if related_query and related_query.strip():
            try:
                related_images = self.searcher.search(related_query)
                self.log(f"找到 {len(related_images)} 张相关图片")
            except Exception as e:
                self.log(f"搜索相关图片时出错: {e}", "ERROR")
        else:
            self.log("无相关图片搜索查询，跳过", "WARNING")
        
        self.log("\n[步骤 3/7] 搜索新元素图片...")
        new_element_query = plan.get("search_new_element_query", "")
        new_element_images = []
        if new_element_query and new_element_query.strip():
            try:
                new_element_images = self.searcher.search(new_element_query)
                self.log(f"找到 {len(new_element_images)} 张新元素图片")
            except Exception as e:
                self.log(f"搜索新元素图片时出错: {e}", "ERROR")
        else:
            self.log("无新元素图片搜索查询，跳过", "WARNING")
        
        all_candidate_images = related_images + new_element_images
        
        if use_original_image and os.path.exists(original_image_path):
            all_candidate_images.insert(0, {
                "url": f"file://{original_image_path}",
                "thumbnail": "",
                "title": "Original Image",
                "source": "Original"
            })
            self.log("已将原始图片加入候选")
        
        if not all_candidate_images:
            self.log("没有找到任何候选图片", "ERROR")
            return None
        
        self.log(f"\n[步骤 4/7] 评估候选图片，寻找可直接使用的（共 {len(all_candidate_images)} 张）...")
        self.log("💡 优化策略：先找可直接使用的图片，避免不必要的分析和生图")
        
        candidate_urls = [img.get("url", "") for img in all_candidate_images if img.get("url")]
        
        if not candidate_urls:
            self.log("没有有效的图片URL", "ERROR")
            return None
        
        edit_instruction = plan.get("edit_instruction", "")
        
        try:
            evaluation_result = self.selector.select_and_evaluate_images(
                image_urls=candidate_urls,
                plan_summary=plan.get("plan_summary", ""),
                question_type=plan.get("question_type", "numerical"),
                edit_instruction=edit_instruction,
                original_image_path=original_image_path,
                image_type=plan.get("image_type", "mixed")  # 传递图片类型
            )
        except Exception as e:
            self.log(f"图片评估时出错: {e}", "ERROR")
            evaluation_result = None
        
        if not evaluation_result:
            self.log("图片评估失败，尝试使用原始图片", "WARNING")
            if os.path.exists(original_image_path):
                try:
                    original_image = Image.open(original_image_path)
                    evaluation_result = {
                        "selected_image": original_image,
                        "can_use_directly": True,
                        "reasoning": "Using original image as fallback"
                    }
                    self.log("使用原始图片作为后备")
                except Exception as e:
                    self.log(f"加载原始图片失败: {e}", "ERROR")
                    return None
            else:
                return None
        
        can_use_directly = evaluation_result.get("can_use_directly", False)
        reasoning = evaluation_result.get("reasoning", "No reasoning provided")
        self.log(f"评估结果: {'✅ 找到可直接使用的图片' if can_use_directly else '❌ 无可直接使用的图片，需要生成'}")
        self.log(f"理由: {reasoning[:200]}...")
        
        if can_use_directly:
            self.log("\n[步骤 4 完成] ⚡ 使用选中的图片，跳过分析和生成（节省成本）")
            final_image = evaluation_result.get("selected_image")
            if final_image is None:
                self.log("选中的图片无效", "ERROR")
                return None
        else:
            self.log("\n[步骤 5/7] 详细分析候选图片并生成融合指令...")
            self.log("💡 因为没有可直接使用的图片，现在开始详细分析")
            
            selected_url = evaluation_result.get("selected_url", "")
            
            selected_images = []
            if selected_url:
                for img in all_candidate_images:
                    if img.get("url") == selected_url:
                        selected_images.append(img)
                        break
            
            if not selected_images:
                selected_images = all_candidate_images[:3]
                self.log("使用前3张候选图片", "WARNING")
            
            try:
                analysis_result = self.analysis_agent.analyze_select_and_plan(
                    images=all_candidate_images,
                    plan_summary=plan.get("plan_summary", ""),
                    edit_instruction=plan.get("edit_instruction", ""),
                    max_selected=2 
                )
                if analysis_result:
                    selected_indices = analysis_result.get("selected_indices", [])
                    fusion_instruction = analysis_result.get("fusion_instruction", "")
                    self.log(f"选中图片索引: {selected_indices}")
                    self.log(f"融合指令: {fusion_instruction[:150]}...")
                    
                    selected_images = [all_candidate_images[idx-1] for idx in selected_indices 
                                     if 1 <= idx <= len(all_candidate_images)]
                else:
                    self.log("图片分析失败，使用默认策略", "WARNING")
                    selected_images = all_candidate_images[:3]
                    fusion_instruction = plan.get("edit_instruction", "Enhance image quality")
                    
            except Exception as e:
                self.log(f"分析图片时出错: {e}", "ERROR")
                selected_images = all_candidate_images[:3]
                fusion_instruction = plan.get("edit_instruction", "Enhance image quality")
                self.log(f"使用默认策略，选择前{len(selected_images)}张", "WARNING")
            
            if not selected_images:
                self.log("没有选出任何图片", "ERROR")
                return None
            
            self.log("\n[步骤 6/7] 🎨 使用融合指令生成新图片...")
            
            if 'fusion_instruction' in locals() and fusion_instruction:
                final_instruction = fusion_instruction
                self.log(f"使用分析Agent生成的融合指令")
            else:
                final_instruction = plan.get("edit_instruction", "Enhance image quality")
                if not final_instruction or not final_instruction.strip():
                    final_instruction = "Enhance the image quality and make it more suitable for questions"
                self.log(f"使用默认编辑指令", "WARNING")
            
            reference_urls = [img.get("url", "") for img in selected_images if img.get("url")]
            reference_urls = [img.get("local_path", img.get("url", "")) for img in selected_images]
            reference_urls = [url if not url.startswith('/') else f"file://{url}" for url in reference_urls if url]
            self.log(f"使用 {len(reference_urls)} 张参考图片生成新图")
            
            try:
                generated_image_urls = self.editor.generate_images(
                    fusion_instruction=final_instruction,
                    reference_image_urls=reference_urls,
                    n=3,
                    image_type=plan.get("image_type", "mixed")  # 传递图片类型
                )
            except Exception as e:
                self.log(f"图片生成时出错: {e}", "ERROR")
                generated_image_urls = []
            
            if not generated_image_urls:
                self.log("图片生成失败，使用评估选中的图片", "WARNING")
                final_image = evaluation_result.get("selected_image")
                if final_image is None:
                    self.log("后备图片也无效", "ERROR")
                    return None
            else:
                self.log(f"生成了 {len(generated_image_urls)} 张新图片")
                
                self.log("\n[步骤 6.5/7] 从生成的图片中选择最佳...")
                try:
                    final_selection = self.selector.select_and_evaluate_images(
                        image_urls=generated_image_urls,
                        plan_summary=plan.get("plan_summary", ""),
                        question_type=plan.get("question_type", "numerical"),
                        image_type=plan.get("image_type", "mixed")  # 传递图片类型
                    )
                except Exception as e:
                    self.log(f"选择最佳图片时出错: {e}", "ERROR")
                    final_selection = None
                
                if not final_selection or not final_selection.get("selected_image"):
                    self.log("从生成图片中选择失败，使用之前选中的图片", "WARNING")
                    final_image = evaluation_result.get("selected_image")
                    if final_image is None:
                        self.log("后备图片也无效", "ERROR")
                        return None
                else:
                    final_image = final_selection.get("selected_image")
                    selected_url = final_selection.get("selected_url", "")
                    self.log(f"最终选择: {selected_url[:100]}...")

        for attempt in range(max_retries + 1):
            self.log(f"\n[步骤 7/7] 生成问题和答案 (尝试 {attempt + 1}/{max_retries + 1})...")
            
            try:
                qa_result = self.generator.generate_question(
                    image_pil=final_image,
                    question_type=plan.get("question_type", "numerical"),
                    original_question=original_question,
                    original_answer=original_answer,
                    plan_summary=plan.get("plan_summary", "")
                )
            except Exception as e:
                self.log(f"问题生成时出错: {e}", "ERROR")
                qa_result = None
            
            if not qa_result:
                self.log(f"问题生成失败 (尝试 {attempt + 1})", "WARNING")
                if attempt < max_retries:
                    continue
                else:
                    return None
            
            question = qa_result.get("question", "")
            answer = qa_result.get("answer", "")
            
            if not question or not answer:
                self.log(f"生成的问题或答案为空 (尝试 {attempt + 1})", "WARNING")
                if attempt < max_retries:
                    continue
                else:
                    return None
            
            self.log(f"生成的问题: {question}")
            self.log(f"答案: {answer}")
            
            self.log("\n[验证] 验证问题质量...")
            try:
                is_valid, reasoning = self.validator.validate(
                    question=question,
                    answer=answer,
                    image_pil=final_image,
                    question_type=plan.get("question_type", "numerical")
                )
            except Exception as e:
                self.log(f"验证时出错: {e}", "ERROR")
                is_valid = False
                reasoning = f"Validation error: {str(e)}"
            
            self.log(f"验证结果: {'通过' if is_valid else '不通过'}")
            self.log(f"验证原因: {reasoning}")
            
            if is_valid:
                try:
                    image_base64 = self.image_to_base64(final_image)
                except Exception as e:
                    self.log(f"图片编码失败: {e}", "ERROR")
                    return None
                
                result = {
                    "question_type": plan.get("question_type", "unknown"),
                    "image_type": plan.get("image_type", "mixed"),  # 🆕 添加 image_type 用于配额管理
                    "question": question,
                    "answer": answer,
                    "image": image_base64
                }
                
                self.log("\n✓ 问题生成成功！", "SUCCESS")
                self.log("="*80)
                return result
            else:
                if attempt < max_retries:
                    self.log(f"验证不通过，重新生成 (剩余重试次数: {max_retries - attempt})", "WARNING")
                else:
                    self.log("达到最大重试次数，放弃该样本", "ERROR")
                    return None
        
        return None
    
    def batch_generate(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        results = []
        total = len(samples) if max_samples is None else min(max_samples, len(samples))
        
        for i, sample in enumerate(samples[:total]):
            self.log(f"\n\n{'='*80}")
            self.log(f"处理样本 {i+1}/{total}")
            
            result = self.generate_question_from_sample(
                original_question=sample.get("original_question", ""),
                original_answer=sample.get("original_answer", ""),
                original_image_path=sample.get("image_path", ""),
                original_question_type=sample.get("original_question_type", "")
            )
            
            if result:
                results.append(result)
                self.log(f"✓ 样本 {i+1} 成功", "SUCCESS")
            else:
                self.log(f"✗ 样本 {i+1} 失败", "ERROR")
                import time
                time.sleep(1.5)
        
        self.log(f"\n\n{'='*80}")
        self.log(f"批量生成完成: 成功 {len(results)}/{total}")
        return results


