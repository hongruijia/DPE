import json
import re
import time
import os
from typing import Dict, List, Optional, Any, Tuple, Callable
from PIL import Image

from api_clients import O3Client, SerperClient, QwenImageEditClient, Qwen3VLClient
from config import (O3_MODEL, QWEN_VL_MODEL, QWEN3_VL_MODEL, MAX_SEARCH_IMAGES, MAX_GENERATED_IMAGES, 
                    MAX_RETRIES, BASE_DELAY, MAX_DELAY, GEMINI_MODEL, CLAUDE_MODEL)
from image_type_strategies import ImageTypeStrategy, ImageType


def retry_on_failure(func: Callable, max_retries: int = MAX_RETRIES, 
                     agent_name: str = "Agent") -> Any:

    for attempt in range(max_retries):
        try:
            result = func()
            
            if result is None:
                print(f"[{agent_name}] 尝试 {attempt + 1}/{max_retries} 返回None，准备重试...")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    print(f"[{agent_name}] 等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[{agent_name}] 已达到最大重试次数，放弃")
                    return None
            else:

                if attempt > 0:
                    print(f"[{agent_name}] 重试成功（尝试 {attempt + 1}/{max_retries}）")
                return result
                
        except Exception as e:
            print(f"[{agent_name}] 尝试 {attempt + 1}/{max_retries} 发生异常")
            print(f"  - 异常类型: {type(e).__name__}")
            print(f"  - 错误信息: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                print(f"[{agent_name}] 等待 {wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"[{agent_name}] 已达到最大重试次数，放弃")
                return None
    
    return None


class PlannerAgent:
    
    def __init__(self, weakness_context: str = ""):
        self.client = O3Client(model_name=CLAUDE_MODEL)
        self.weakness_context = weakness_context
    
    def create_plan(self, original_question: str, original_answer: str, 
                   image_path: str, original_question_type: str = "",
                   preset_image_type: str = "") -> Optional[Dict[str, Any]]:

        original_type_info = ""
        if original_question_type:
            original_type_info = f"\nOriginal Question Type: {original_question_type} (for reference)"
        
        weakness_info = ""
        if self.weakness_context:
            weakness_info = f"\n\nIMPORTANT - Focus Areas (based on model weaknesses):\n{self.weakness_context}\n\nPlease prioritize generating questions that address these weaknesses."
        
        if preset_image_type:
            preset_type_info = f"\n\nIMPORTANT: The image type is PRESET to '{preset_image_type}'. You MUST generate a question that fits this type. DO NOT generate the 'image_type' field."
            
            prompt = f"""You are an intelligent Question Generation Planner. Given an original VQA question and image, create a plan to generate a NEW, challenging question.

Original Question: {original_question}
Original Answer: {original_answer}{original_type_info}{weakness_info}{preset_type_info}

Your task:
1. Analyze the original question and image carefully
2. Decide what type of NEW question to generate (choose ONE from: multiple choice, numerical, regression)
3. Plan how to create or modify an image to support the new question (MUST fit '{preset_image_type}' type)
4. Provide search queries for:
   - Related images (similar context to original)
   - New element images (to introduce new concepts/objects)
5. Provide a BASE instruction for image editing/generation

Output MUST be in valid JSON format:
{{
    "question_type": "multiple choice|numerical|regression",
    "plan_summary": "Brief summary of your plan (what kind of question you'll generate and why)",
    "search_related_query": "Search query for related images (these will provide context similar to the original)",
    "search_new_element_query": "Search query for new element images (these will introduce new concepts)",
    "edit_instruction": "Base instruction describing WHAT to modify/create and WHAT elements to combine from the searched images. Focus on the goal (e.g., 'add red sports car to the street scene', 'change sky to sunset colors', 'add mountain background'). Later, this will be expanded into detailed per-image instructions."
}}

Requirements:
- The new question should be DIFFERENT but related to the original
- Make it challenging and require reasoning
- Be specific in your search queries
- In edit_instruction, describe WHAT to achieve (the goal), not HOW to do it technically
- The edit_instruction will be used as a base to generate detailed per-image instructions later
- Output ONLY the JSON, no extra text
- DO NOT include 'image_type' field (it's preset to '{preset_image_type}')

Example edit_instruction:
- Good: "Add a red sports car to the street scene and change the sky to dramatic sunset colors"
- Good: "Replace the background with mountain landscape and add autumn foliage colors"
- Bad: "Modify the image" (too vague)
- Bad: "Use Image 1 as base..." (too technical, this will be added automatically later)"""
        else:
            prompt = f"""You are an intelligent Question Generation Planner. Given an original VQA question and image, create a plan to generate a NEW, challenging question.

Original Question: {original_question}
Original Answer: {original_answer}{original_type_info}{weakness_info}

Your task:
1. Analyze the original question and image carefully
2. Decide what type of NEW question to generate (choose ONE from: multiple choice, numerical, regression)
3. Identify the IMAGE TYPE based on the question content (see image type categories below)
4. Plan how to create or modify an image to support the new question
5. Provide search queries for:
   - Related images (similar context to original)
   - New element images (to introduce new concepts/objects)
6. Provide a BASE instruction for image editing/generation

IMAGE TYPE CATEGORIES (choose ONE that best fits):
- geometric: Geometric shapes, angles, triangles, circles, polygons (requires extreme precision)
- medical: Medical images, X-rays, CT scans, anatomical diagrams
- chart_graph: Charts, graphs, data visualizations, statistics
- text_dense: Text-heavy documents, receipts, menus, OCR content
- diagram: Flowcharts, schematics, circuit diagrams, process diagrams
- math_formula: Mathematical formulas, equations, expressions
- map: Maps, geographic locations, routes, navigation
- natural_scene: Natural landscapes, outdoor scenes, nature
- artistic: Artistic images, paintings, creative designs
- everyday_object: Common objects, products, furniture, household items
- architectural: Buildings, structures, architectural designs
- mixed: Mixed or unclear category

Output MUST be in valid JSON format:
{{
    "question_type": "multiple choice|numerical|regression",
    "image_type": "geometric|medical|chart_graph|text_dense|diagram|math_formula|map|natural_scene|artistic|everyday_object|architectural|mixed",
    "plan_summary": "Brief summary of your plan (what kind of question you'll generate and why)",
    "search_related_query": "Search query for related images (these will provide context similar to the original)",
    "search_new_element_query": "Search query for new element images (these will introduce new concepts)",
    "edit_instruction": "Base instruction describing WHAT to modify/create and WHAT elements to combine from the searched images. Focus on the goal (e.g., 'add red sports car to the street scene', 'change sky to sunset colors', 'add mountain background'). Later, this will be expanded into detailed per-image instructions."
}}

Requirements:
- The new question should be DIFFERENT but related to the original
- Make it challenging and require reasoning
- Be specific in your search queries
- In edit_instruction, describe WHAT to achieve (the goal), not HOW to do it technically
- The edit_instruction will be used as a base to generate detailed per-image instructions later
- Output ONLY the JSON, no extra text

Example edit_instruction:
- Good: "Add a red sports car to the street scene and change the sky to dramatic sunset colors"
- Good: "Replace the background with mountain landscape and add autumn foliage colors"
- Bad: "Modify the image" (too vague)
- Bad: "Use Image 1 as base..." (too technical, this will be added automatically later)"""

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    print(f"[PlannerAgent] 等待 {wait_time}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat(messages, image_path=image_path, max_tokens=8192)
                
                print(f"[PlannerAgent API Response]:\n{response}\n")
                
                if not response:
                    print(f"[PlannerAgent] API返回None，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    print(f"[PlannerAgent] 错误: 无法从响应中提取JSON")
                    print(f"  - 响应内容: {response}")
                    continue
                
                json_str = json_match.group()
                json_str = re.sub(r',(\s*)}', r'\1}', json_str)  # 移除 },} 中的逗号
                json_str = re.sub(r',(\s*)]', r'\1]', json_str)  # 移除 ],] 中的逗号
                
                plan = json.loads(json_str)
                
                plan = self._validate_and_fix_plan(plan, preset_image_type)
                
                if plan:
                    if preset_image_type:
                        print(f"[PlannerAgent] 🎯 使用预设 image_type: {preset_image_type}")
                        plan["image_type"] = preset_image_type
                        image_type = ImageType(preset_image_type)  # 🔧 定义 image_type 变量
                    else:
                        image_type_str = plan.get("image_type", "")
                        
                        if image_type_str:
                            try:
                                image_type = ImageType(image_type_str)
                                print(f"[PlannerAgent] ✓ LLM 返回 image_type: {image_type.value}")
                                plan["image_type"] = image_type.value
                            except ValueError:
                                print(f"[PlannerAgent] ⚠️  无效的 image_type: {image_type_str}, 使用 GPT-4o 重新识别")
                                image_type = ImageTypeStrategy.identify_image_type(
                                    plan_summary=plan.get("plan_summary", ""),
                                    question_type=plan.get("question_type", ""),
                                    original_question=original_question,
                                    use_gpt4o=True  
                                )
                                plan["image_type"] = image_type.value
                        else:
                            print(f"[PlannerAgent] ⚠️  LLM 未返回 image_type，使用 GPT-4o 识别")
                            image_type = ImageTypeStrategy.identify_image_type(
                                plan_summary=plan.get("plan_summary", ""),
                                question_type=plan.get("question_type", ""),
                                original_question=original_question,
                                use_gpt4o=True  
                            )
                            plan["image_type"] = image_type.value
                    
                    strategy = ImageTypeStrategy.get_strategy(image_type)
                    print(f"[PlannerAgent] 📸 图片类型: {image_type.value}")
                    print(f"  - {strategy['description']}")
                    print(f"  - 策略: {strategy['priority']}, 编辑范围: {strategy['edit_scope']}")
                    
                    print(f"[PlannerAgent] ✓ 计划验证通过")
                    if attempt > 0:
                        print(f"[PlannerAgent] ✓ 重试成功（尝试 {attempt + 1}/{MAX_RETRIES}）")
                    return plan
                else:
                    print(f"[PlannerAgent] 计划验证失败，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"[PlannerAgent] JSON解析失败，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 错误: {e}")
                print(f"  - 响应内容: {response if 'response' in locals() else 'N/A'}")
                continue
            except Exception as e:
                print(f"[PlannerAgent] 异常，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 异常类型: {type(e).__name__}")
                print(f"  - 错误信息: {str(e)}")
                continue
        
        print(f"[PlannerAgent] ❌ 所有重试失败，已尝试 {MAX_RETRIES} 次")
        return None
    
    def _validate_and_fix_plan(self, plan: Dict[str, Any], preset_image_type: str = "") -> Optional[Dict[str, Any]]:

        required_fields = {
            "question_type": "numerical",  
            "image_type": preset_image_type or "mixed",  
            "plan_summary": "Generate a challenging question based on the image",
            "search_related_query": "similar image",
            "search_new_element_query": "related objects",
            "edit_instruction": "Enhance the image quality and clarity"
        }
        
        if "question_type" not in plan or not plan["question_type"]:
            print(f"[PlannerAgent] 警告: 缺少question_type字段，使用默认值")
            plan["question_type"] = required_fields["question_type"]
        
        valid_types = ["multiple choice", "numerical", "regression"]
        if plan["question_type"] not in valid_types:
            print(f"[PlannerAgent] 警告: question_type '{plan['question_type']}' 无效，使用默认值")
            plan["question_type"] = required_fields["question_type"]
        
        if preset_image_type:
            plan["image_type"] = preset_image_type
            print(f"[PlannerAgent] 使用预设 image_type: {preset_image_type}")
        else:
            valid_image_types = [
                "geometric", "medical", "chart_graph", "text_dense", "diagram",
                "math_formula", "map", "natural_scene", "artistic", "everyday_object",
                "architectural", "mixed"
            ]
            if "image_type" not in plan or not plan["image_type"]:
                print(f"[PlannerAgent] 警告: 缺少image_type字段，使用默认值")
                plan["image_type"] = required_fields["image_type"]
            elif plan["image_type"] not in valid_image_types:
                print(f"[PlannerAgent] 警告: image_type '{plan['image_type']}' 无效，使用默认值")
                plan["image_type"] = required_fields["image_type"]
        
        for field, default_value in required_fields.items():
            if field == "image_type":
                continue  # 已经处理过了
            if field not in plan or not plan[field]:
                print(f"[PlannerAgent] 警告: 缺少{field}字段，使用默认值: {default_value}")
                plan[field] = default_value
        
        print(f"[PlannerAgent] 字段验证完成: {list(plan.keys())}")
        return plan


class ImageSearchAgent:
    
    def __init__(self):
        self.client = SerperClient()
    
    def search(self, query: str, num_results: int = MAX_SEARCH_IMAGES) -> List[Dict[str, str]]:
        
        return self.client.search_images(query, num_results=num_results)


class ImageAnalysisAndSelectionAgent:
    
    def __init__(self):
        self.client = O3Client(model_name=GEMINI_MODEL)
    
    def analyze_select_and_plan(self, images: List[Dict[str, str]], plan_summary: str,
                                edit_instruction: str, max_selected: int = 3) -> Optional[Dict[str, Any]]:
        
        if not images:
            return None
        
        print(f"\n{'='*80}")
        print(f"[ImageAnalysisAgent] 🔍 阶段1：批量分析候选图片（真正看图 - 策略B）")
        print(f"{'='*80}\n")
        
        images_to_analyze = images[:min(len(images), 6)]  # 最多分析6张候选
        real_descriptions = {}
        
        batch_size = 3
        for batch_start in range(0, len(images_to_analyze), batch_size):
            batch_end = min(batch_start + batch_size, len(images_to_analyze))
            batch_images = images_to_analyze[batch_start:batch_end]
            
            print(f"\n--- 批量分析第 {batch_start//batch_size + 1} 批 ({batch_end - batch_start}张图片) ---")
            
            batch_descriptions = self._analyze_images_batch(batch_images, batch_start + 1)
            
            if batch_descriptions:
                real_descriptions.update(batch_descriptions)
                for idx, desc in batch_descriptions.items():
                    print(f"  ✓ 图片{idx}: {desc[:80]}...")
            else:
                print(f"  ⚠️  批量分析失败，降级为逐个分析...")
                for idx, img in enumerate(batch_images, batch_start + 1):
                    img_url = img.get('url', '')
                    img_path = img.get('local_path', '')
                    print(f"  - 分析图片 {idx}...")
                    description = self._analyze_single_image_visual(img_url, img_path, idx)
                    if description:
                        real_descriptions[str(idx)] = description
                        print(f"    ✓ {description[:60]}...")
                    else:
                        real_descriptions[str(idx)] = f"[图片{idx}基于标题: {img.get('title', 'Unknown')}]"
                        print(f"    ⚠️  使用标题fallback")
                    if idx < len(images_to_analyze):
                        time.sleep(0.5)
            
            if batch_end < len(images_to_analyze):
                time.sleep(1.5)
        
        print(f"\n{'='*80}")
        print(f"[ImageAnalysisAgent] 📋 阶段2：基于真实视觉内容选择最佳组合并生成融合指令")
        print(f"{'='*80}\n")
        
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    print(f"[ImageAnalysisAgent] 等待 {wait_time}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                
                descriptions_text = "\n\n".join([
                    f"Image {idx}:\n  Visual Content: {desc}\n  Title: {images_to_analyze[int(idx)-1].get('title', 'N/A')}\n  Source: {images_to_analyze[int(idx)-1].get('source', 'N/A')}"
                    for idx, desc in real_descriptions.items()
                ])
                
                prompt = f"""You are an Expert Image Fusion Planner with REAL visual descriptions of candidate images.

CONTEXT:
- Plan Goal: {plan_summary}
- Edit Instruction: {edit_instruction}

CANDIDATE IMAGES (with REAL visual analysis):
{descriptions_text}

YOUR TASK:
1. SELECT 1-2 images (NOT 3!) that work best for a SIMPLE editing operation
2. GENERATE a DETAILED, PRECISE fusion instruction following these principles:
   - Operation should be SIMPLE (one edit or one addition)
   - Description should be DETAILED (specific parameters, exact positions, clear effects)

🎯 DETAILED INSTRUCTION REQUIREMENTS (Simple Operation, Detailed Description):

STRATEGY A: SINGLE IMAGE EDIT (Preferred - Select only 1 image)
   ✅ GOOD Examples (Detailed & Specific):
     * "Edit Image 1: Increase the contrast by 25% and brightness by 15% for the three red apples in the center of the image to make them stand out, while keeping the green background unchanged to maintain depth"
     * "Edit Image 2: Apply a Gaussian blur with radius 8px to the background buildings behind the person, leaving the person in sharp focus to create a portrait depth-of-field effect"
     * "Edit Image 1: Increase color saturation by 30% for the blue car in the foreground, making its metallic paint more vibrant, while keeping the sky and road colors natural"
     * "Edit Image 3: Rotate the document 90 degrees clockwise and increase sharpness by 40% to make all text clearly readable, especially the small print at the bottom"
   
   ❌ BAD Examples (Too Vague):
     * "Edit Image 1: Enhance the contrast and brightness to make the objects more visible" ← Which objects? How much? What about other elements?
     * "Edit Image 2: Add a slight blur to the background" ← How much blur? Which part of background? What's the desired effect?

STRATEGY B: SIMPLE TWO-IMAGE ADDITION (Only if absolutely necessary - Select 2 images)
   ✅ GOOD Examples (Detailed & Specific):
     * "Use Image 1 as the base (city street scene). Add the red sports car from Image 2, positioned on the right lane of the street, facing left (towards the camera), scaled to approximately 15% of the image width, placed on the asphalt road surface near the traffic light"
     * "Use Image 2 as the base (living room interior). Place the large green potted plant from Image 1 in the bottom-left corner next to the sofa, scaled to about 1.2 meters tall (20% of image height), ensuring it sits on the wooden floor"
     * "Use Image 1 as the base (mountain landscape). Add the snow-capped mountain peak from Image 3 in the background, centered horizontally, scaled to fill the upper 40% of the sky area, creating a dramatic backdrop"
   
   ❌ BAD Examples (Too Vague):
     * "Use Image 1 as the base. Add the red car from Image 2 to the center of the scene" ← Which direction? How big? On road or grass? What perspective?
     * "Use Image 2 as the base. Place the mountain from Image 1 in the background" ← Where exactly? How big? What altitude?

📋 REQUIRED DETAILS FOR IMAGE GENERATION AI:

🎨 For SINGLE IMAGE EDIT, provide these specifics:
   
   1. **Target Identification** (What to edit):
      ✓ "the three red spheres in the upper-left quadrant"
      ✗ "the objects" or "some items"
   
   2. **Action Verbs** (Clear operations for AI):
      ✓ Use: increase, decrease, adjust, enhance, saturate, brighten, darken, sharpen, blur, rotate, scale
      ✗ Avoid: improve, make better, fix, enhance (too vague)
   
   3. **Numeric Parameters** (Precise values):
      ✓ "increase brightness by 20%", "rotate 45 degrees clockwise"
      ✓ "adjust saturation from 0.5 to 0.8", "blur radius 5px"
      ✗ "make brighter", "rotate a bit"
   
   4. **Visual Properties** (Colors, textures, materials):
      ✓ "the metallic blue surface", "the rough wooden texture"
      ✓ "RGB(255,0,0) for the highlights"
      ✗ "the nice looking part"
   
   5. **Preservation Instructions** (What stays unchanged):
      ✓ "while maintaining the white background and grid lines at their original opacity"
      ✓ "keeping all text labels and axes unchanged"
      ✗ "don't change other stuff"
   
   6. **Lighting & Atmosphere** (If relevant):
      ✓ "under soft daylight lighting", "with subtle shadow cast to the right"
      ✓ "maintaining the warm indoor lighting tone"
   
   7. **Desired Visual Outcome**:
      ✓ "to create a clear visual hierarchy with the modified element as focal point"
      ✓ "resulting in a high-contrast distinction for easy identification"

🔗 For TWO-IMAGE ADDITION, provide these specifics:
   
   1. **Base Context** (Scene description):
      ✓ "urban street scene with asphalt road, traffic lights, and modern buildings in background"
      ✗ "street scene"
   
   2. **Object Description** (Full visual details):
      ✓ "the glossy red sports car with black racing stripes, approximately 4.5m long"
      ✗ "the car"
   
   3. **Spatial Positioning** (Precise 3D placement):
      ✓ "positioned in the right traffic lane, 15 meters from the foreground"
      ✓ "at coordinates (65%, 40%) from top-left, occupying 15% of image width"
      ✗ "in the middle somewhere"
   
   4. **Orientation & Perspective**:
      ✓ "facing 30 degrees left from camera view, slightly angled downward"
      ✓ "side profile view showing the left side of the object"
      ✗ "facing left"
   
   5. **Scale & Proportions**:
      ✓ "scaled to 12% of total image width (approximately 180 pixels wide)"
      ✓ "matching the perspective scale of surrounding buildings (1.5m tall at that distance)"
      ✗ "medium sized"
   
   6. **Surface & Ground Plane**:
      ✓ "placed on the gray asphalt road surface, with wheels touching ground plane"
      ✓ "sitting on the wooden floor, aligned with the floor perspective grid"
      ✗ "on the ground"
   
   7. **Integration Details** (Realism):
      ✓ "with appropriate shadow cast toward bottom-right matching scene lighting"
      ✓ "color-graded to match the scene's warm afternoon sunlight tone"
      ✓ "with slight motion blur on wheels suggesting forward movement"
   
   8. **Spatial Relationships**:
      ✓ "positioned between the first traffic light pole (left) and the crosswalk marking (right)"
      ✓ "approximately 3 car lengths behind the white sedan in foreground"
      ✗ "near some stuff"

🤖 AI-FRIENDLY INSTRUCTION STRUCTURE:

Use this template for clarity:
```
[ACTION] [TARGET] [PARAMETERS] [PRESERVATION] [OUTCOME]

Example:
"Increase the contrast (by 30%) and saturation (by 25%) of the three blue triangles 
in the center of the diagram, making their edges sharp and colors vivid, while keeping 
the white background, black axis lines, and all text labels at original values, 
resulting in a clear visual distinction between the triangles and background for 
easy shape identification."
```

⚠️ CRITICAL CONSTRAINTS:
   - NEVER select more than 2 images
   - AVOID complex multi-element fusion
   - ONE simple operation (edit OR add), but described in FULL detail
   - Include specific numbers/percentages where applicable
   - Describe spatial relationships and context clearly
   - Make it actionable - a human or AI should be able to execute it precisely

FORMAT EXAMPLES (AI-Optimized):

Example 1 (Single edit - Geometric precision):
{{
    "selected_indices": [1],
    "selection_reasoning": "Image 1 contains the geometric shapes needed but lacks visual clarity for precise angle measurement questions",
    "fusion_instruction": "Increase the stroke width of the three overlapping circles in the center from 2px to 4px, and adjust their colors to high-contrast values (circle 1: RGB(255,0,0) red, circle 2: RGB(0,0,255) blue, circle 3: RGB(0,200,0) green), while maintaining the white background (RGB(255,255,255)) and black coordinate axes unchanged. Apply 30% brightness increase to the intersection regions only, creating clearly visible overlapping areas with distinct colors (purple, cyan, yellow) for easy visual identification. Keep all text labels (angle markers α, β, γ) and grid lines at their original size and position."
}}

Example 2 (Two-image addition - Photorealistic scene):
{{
    "selected_indices": [2, 3],
    "selection_reasoning": "Image 2 provides an ideal urban street base scene, Image 3 contains the specific vehicle model needed for counting questions",
    "fusion_instruction": "Use Image 2 as the base scene (urban intersection with asphalt road, white lane markings, traffic light pole on left at 20% from left edge, modern glass buildings in background under overcast sky lighting). Extract the red delivery truck from Image 3 (metallic red paint, white cargo box, approximately 6m long × 2.5m wide × 3m tall). Place the truck in the right traffic lane at position 60% from left edge and 45% from top, scaled to 180 pixels wide to match perspective depth of 30 meters from camera. Orient the truck facing toward bottom-left (heading angle 215 degrees), showing 3/4 front-left view. Position all four wheels firmly on the gray asphalt road surface, aligned with the road's perspective vanishing point. Add a soft shadow beneath the vehicle (opacity 40%, blur radius 8px) cast toward bottom-right to match the overcast lighting. Color-grade the truck to match the scene's cool gray color temperature. Position between the first traffic light (on left) and the crosswalk zebra stripes (15m ahead), maintaining realistic traffic spacing."
}}

Example 3 (Single edit - Medical imaging):
{{
    "selected_indices": [4],
    "selection_reasoning": "Image 4 has the correct CT scan but tissue contrast is too low for inflammation identification questions",
    "fusion_instruction": "Apply targeted contrast enhancement to the tissue region in the upper-right quadrant (coordinates 70%-90% from left, 10%-30% from top). Increase the red channel intensity by 40% (from base value ~120 to ~168) for the inflammation markers, while keeping the surrounding healthy tissue at original grayscale values (RGB ~80,80,80). Adjust the local brightness in a 50-pixel radius around the inflammation center by +25%, creating a gradual falloff (Gaussian gradient) to the normal tissue brightness. Maintain all DICOM metadata overlays (patient info, scan parameters in white text at corners), the measurement scale bar (bottom-right, 10mm calibration), and the grayscale lookup table unchanged. Apply a subtle sharpening filter (unsharp mask, radius 1.5px, amount 60%) to the enhanced region only, making tissue boundaries and inflammation edges more clearly defined while keeping other anatomical structures at their original sharpness for diagnostic accuracy."
}}

Example 4 (Two-image addition - Indoor scene with object):
{{
    "selected_indices": [1, 5],
    "selection_reasoning": "Image 1 has perfect living room setup, Image 5 contains the decorative plant needed for spatial reasoning questions",
    "fusion_instruction": "Use Image 1 as the base (modern living room interior: wooden floor with natural grain texture, gray L-shaped sofa on right, white walls, soft warm lighting from ceiling lamps creating color temperature ~3200K). Extract the large green potted plant from Image 5 (Monstera deliciosa with 8 large leaves, dark green color RGB(34,90,49), in white ceramic pot 40cm diameter). Place the plant in the bottom-left corner at coordinates (8%, 75%) from top-left origin. Scale the plant to 160 pixels height (representing ~1.2m tall in real space) to maintain proper proportion with the sofa (2m wide). Ensure the pot base sits flat on the wooden floor planks, aligned with the floor's perspective grid lines converging toward the room's back wall. Position the plant 15cm from the left wall edge and 25cm from the sofa's left armrest, creating a cozy corner arrangement. Cast a soft shadow on the floor extending toward bottom-right (shadow opacity 25%, blur radius 12px, length ~60cm) matching the warm overhead lighting direction. Apply slight color temperature adjustment to the plant (+150K warmer) to blend with the room's ambient lighting. Add subtle ambient occlusion darkening (10% brightness reduction) where the pot meets the floor for realistic ground contact. Ensure leaves naturally overlap the white wall background with proper depth layering, and the plant appears behind the sofa arm (z-order: wall < plant < sofa)."
}}

OUTPUT FORMAT (JSON ONLY):
{{
    "selected_indices": [1] or [1, 2],
    "selection_reasoning": "Explain why these specific image(s) are optimal for the task",
    "fusion_instruction": "One DETAILED AI-executable instruction (3-6 sentences) with ALL parameters needed for perfect generation"
}}

🤖 WRITING FOR IMAGE GENERATION AI:

Think of yourself as writing code for an image editing API. The AI needs:

✅ DO:
- Use computer vision terminology (RGB values, pixel coordinates, percentages)
- Specify exact positions using coordinates or percentages
- Include lighting/shadow details for realism
- Mention z-order/layering when elements overlap
- Describe colors in RGB, HSV, or specific names
- Give sizes in pixels, percentages, or real-world units with scale context
- Use precise action verbs (increase, adjust, place, scale, rotate)
- Include integration details (shadows, reflections, color grading)

✗ DON'T:
- Use subjective terms ("beautiful", "nice", "good-looking")
- Give ambiguous directions ("somewhere in the middle", "a bit more")
- Skip lighting/shadow information
- Forget about depth/perspective/scale relationships
- Use metaphors or analogies
- Reference abstract concepts without visual specifics

📐 PRECISION CHECKLIST:

Before finalizing your instruction, verify it includes:
□ Exact numerical parameters (percentages, pixels, degrees)
□ Precise color specifications (RGB values or specific color names)
□ Absolute or relative positioning (coordinates, percentages, or spatial relationships)
□ Scale/size with units (pixels, percentages, or real-world dimensions with context)
□ Orientation/rotation (degrees, cardinal directions, or view angles)
□ Surface/plane specification (what the object sits on or attaches to)
□ Lighting/shadow direction and intensity (if adding objects)
□ Preservation instructions (what must NOT change)
□ Visual outcome description (what the result should look like)

REMEMBER: 
- Operation: SIMPLE (one edit or one addition)
- Description: HYPER-DETAILED (AI needs every visual parameter)
- Format: AI-EXECUTABLE (like writing API parameters)
- The AI cannot guess or infer - spell out EVERYTHING visually relevant
"""
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat(messages, max_tokens=8192)
                
                print(f"[ImageAnalysisAgent 选择与规划 Response]:\n{response}\n")
                
                if not response:
                    print(f"[ImageAnalysisAgent] API返回None，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    print(f"[ImageAnalysisAgent] 错误: 响应中未找到JSON，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                json_str = json_match.group()
                json_str = re.sub(r',(\s*)}', r'\1}', json_str)  # 移除 },} 中的逗号
                json_str = re.sub(r',(\s*)]', r'\1]', json_str)  # 移除 ],] 中的逗号
                
                result = json.loads(json_str)
                
                if "selected_indices" not in result or not result["selected_indices"]:
                    print(f"[ImageAnalysisAgent] 错误: 缺少selected_indices，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                if "fusion_instruction" not in result or len(result["fusion_instruction"]) < 20:
                    print(f"[ImageAnalysisAgent] 错误: fusion_instruction太短或缺失，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                selected_indices = result["selected_indices"]
                valid_indices = [idx for idx in selected_indices if isinstance(idx, int) and 1 <= idx <= len(images_to_analyze)]
                
                if len(valid_indices) == 0:
                    print(f"[ImageAnalysisAgent] 错误: 没有有效索引，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                result["selected_indices"] = valid_indices[:2]  # 最多2张
                
                if len(valid_indices) > 2:
                    print(f"[ImageAnalysisAgent] ⚠️  选择了{len(valid_indices)}张图片，截断为前2张（简化策略）")
                result["image_descriptions"] = {idx: real_descriptions[idx] for idx in map(str, result["selected_indices"]) if idx in real_descriptions}
                
                print(f"\n{'='*80}")
                print(f"[ImageAnalysisAgent] ✅ 分析与规划完成！")
                print(f"  - 选中图片: {result['selected_indices']}")
                print(f"  - 融合指令长度: {len(result['fusion_instruction'])} 字符")
                print(f"  - 融合指令预览: {result['fusion_instruction'][:150]}...")
                print(f"{'='*80}\n")
                
                if attempt > 0:
                    print(f"[ImageAnalysisAgent] ✓ 重试成功（尝试 {attempt + 1}/{MAX_RETRIES}）")
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"[ImageAnalysisAgent] JSON解析失败，尝试 {attempt + 1}/{MAX_RETRIES}: {e}")
                continue
            except Exception as e:
                print(f"[ImageAnalysisAgent] 异常，尝试 {attempt + 1}/{MAX_RETRIES}: {type(e).__name__} - {str(e)}")
                continue
        
        # Fallback
        print(f"[ImageAnalysisAgent] ❌ 所有重试失败，使用fallback（简化策略：单图编辑）")
        return {
            "selected_indices": [1],  
            "image_descriptions": real_descriptions,
            "fusion_instruction": f"Edit Image 1: {edit_instruction if edit_instruction else 'Enhance the image quality and clarity'}"
        }
    
    def _analyze_images_batch(self, images: List[Dict[str, str]], start_index: int) -> Optional[Dict[str, str]]:

        if not images or len(images) == 0:
            return None
        
        image_urls_list = []
        indices = []
        
        for idx, img in enumerate(images, start_index):
            url = img.get('url', '')
            if url:
                image_urls_list.append(url)
                indices.append(idx)
        
        if not image_urls_list:
            return None
        
        images_info = "\n".join([
            f"Image {idx}: Title='{images[idx-start_index].get('title', 'N/A')}', Source='{images[idx-start_index].get('source', 'N/A')}'"
            for idx in indices
        ])
        
        prompt = f"""You are viewing {len(image_urls_list)} candidate images. Analyze EACH image and provide detailed visual descriptions.

Images to analyze:
{images_info}

For EACH image, provide:
1. Main Subject: Primary focus (objects, people, scenes)
2. Colors: Dominant colors, palette, mood
3. Composition: Layout, element positioning
4. Style & Quality: Visual style, clarity, professional quality
5. Key Details: Elements useful for image editing/fusion

Output format (JSON):
{{
    "descriptions": {{
        "{indices[0]}": "Detailed paragraph about image {indices[0]} (3-5 sentences)",
        {f'"{indices[1]}": "Detailed paragraph about image {indices[1]} (3-5 sentences)",' if len(indices) > 1 else ''}
        {f'"{indices[2]}": "Detailed paragraph about image {indices[2]} (3-5 sentences)"' if len(indices) > 2 else ''}
    }}
}}

Be specific, factual, and focus on visual elements that can be extracted or integrated."""
        
        try:
            print(f"    [批量分析] 准备传入 {len(image_urls_list)} 张图片...")
            for i, url in enumerate(image_urls_list, 1):
                print(f"      图片{i}: {url[:80]}...")
            
            messages = [{"role": "user", "content": prompt}]
            
            print(f"    [批量分析] 调用API中（预计30-90秒，请耐心等待）...")
            import time
            start_time = time.time()
            
            try:
                response = self.client.chat(messages, image_urls=image_urls_list, max_tokens=4096)
                elapsed_time = time.time() - start_time
                print(f"    [批量分析] API调用完成，耗时 {elapsed_time:.1f}秒")
            except Exception as api_error:
                elapsed_time = time.time() - start_time
                print(f"    ✗ API调用异常（耗时 {elapsed_time:.1f}秒）")
                print(f"    ✗ 错误类型: {type(api_error).__name__}")
                print(f"    ✗ 错误信息: {str(api_error)[:200]}")
                return None
            
            if not response:
                print(f"    ✗ API返回None（可能超时或被拒绝）")
                return None
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                print(f"    ✗ 批量分析响应无JSON格式")
                return None
            
            json_str = json_match.group()
            json_str = re.sub(r',(\s*)}', r'\1}', json_str)  # 移除 },} 中的逗号
            json_str = re.sub(r',(\s*)]', r'\1]', json_str)  # 移除 ],] 中的逗号
            
            result = json.loads(json_str)
            descriptions_dict = result.get("descriptions", {})
            
            valid_descriptions = {}
            for idx in indices:
                desc = descriptions_dict.get(str(idx), "")
                if desc and len(desc) > 50:
                    valid_descriptions[str(idx)] = desc.strip()
            
            if len(valid_descriptions) > 0:
                return valid_descriptions
            else:
                return None
                
        except Exception as e:
            print(f"    ✗ 批量分析异常: {type(e).__name__} - {str(e)}")
            return None
    
    def _analyze_single_image_visual(self, image_url: str, image_path: str, index: int) -> Optional[str]:

        prompt = """Analyze this image and provide a DETAILED visual description covering:

1. Main Subject: What is the primary focus? (objects, people, scenes)
2. Colors: Dominant colors, color palette, mood
3. Composition: Layout, positioning of elements, foreground/background
4. Style: Photography style, artistic style, quality indicators
5. Details: Important visual details that could be useful for image editing

Be specific and factual. Focus on visual elements that can be extracted, modified, or integrated with other images.

Output format: A single paragraph (3-5 sentences) covering all points above."""

        try:
            messages = [{"role": "user", "content": prompt}]
            
            if image_url.startswith("http://") or image_url.startswith("https://"):
                response = self.client.chat(messages, image_url=image_url, max_tokens=2048)
            else:
                local_path = image_path if image_path else image_url.replace("file://", "")
                if not os.path.exists(local_path):
                    print(f"  ✗ 本地文件不存在: {local_path}")
                    return None
                response = self.client.chat(messages, image_path=local_path, max_tokens=2048)
            
            if response and len(response.strip()) > 50:
                return response.strip()
            else:
                return None
                
        except Exception as e:
            print(f"  ✗ 分析图片{index}异常: {type(e).__name__} - {str(e)}")
            return None


class ImageEditAgent:
    
    def __init__(self):
        self.client = QwenImageEditClient()
    
    def generate_images(self, fusion_instruction: str, reference_image_urls: List[str],
                       n: int = MAX_GENERATED_IMAGES, image_type: str = "mixed") -> List[str]:

        try:
            img_type_enum = ImageType(image_type)
            formatted_instruction = ImageTypeStrategy.format_edit_instruction(
                img_type_enum, fusion_instruction
            )
            print(f"\n{'='*80}")
            print(f"[ImageEditAgent] 🎨 开始生成融合图片")
            print(f"  图片类型: {image_type.upper()}")
            print(f"{'='*80}")
            print(f"  参考图片数量: {len(reference_image_urls)}")
            print(f"  生成数量: {n}")
            print(f"\n  原始指令:")
            print(f"  {fusion_instruction}")
            print(f"\n  📋 格式化后的类型特定指令:")
            print(f"  {formatted_instruction[:300]}...")
            print()
        except ValueError:
            formatted_instruction = fusion_instruction
            print(f"\n{'='*80}")
            print(f"[ImageEditAgent] 🎨 开始生成融合图片")
            print(f"{'='*80}")
            print(f"  参考图片数量: {len(reference_image_urls)}")
            print(f"  生成数量: {n}")
            print(f"\n  融合指令:")
            print(f"  {fusion_instruction}\n")
        
        for attempt in range(3):  
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt * 2
                    print(f"[ImageEditAgent] 重试 ({attempt + 1}/3)，等待 {wait_time}秒...")
                    time.sleep(wait_time)
                
                generated_urls = self.client.edit_images(
                    instruction=formatted_instruction,
                    reference_images=reference_image_urls,
                    n=min(n, 6),  
                    prompt_extend=True  
                )
                
                if generated_urls and len(generated_urls) > 0:
                    print(f"\n[ImageEditAgent] ✅ 成功生成 {len(generated_urls)} 张图片")
                    for idx, url in enumerate(generated_urls, 1):
                        print(f"  {idx}. {url[:80]}...")
                    print(f"{'='*80}\n")
                    return generated_urls
                else:
                    print(f"[ImageEditAgent] ⚠️  尝试 {attempt + 1}/3: 生成失败，返回空列表")
                    continue
                    
            except Exception as e:
                print(f"[ImageEditAgent] ❌ 尝试 {attempt + 1}/3 异常: {type(e).__name__} - {str(e)}")
                continue
        
        print(f"[ImageEditAgent] ❌ 所有重试失败，无法生成图片")
        return []


class ImageSelectorAgent:
    
    def __init__(self):
        self.client = Qwen3VLClient(model_name=QWEN3_VL_MODEL)
        self.image_edit_client = QwenImageEditClient()
    
    def select_and_evaluate_images(self, image_urls: List[str], plan_summary: str,
                                   question_type: str, edit_instruction: str = "",
                                   original_image_path: str = None, 
                                   image_type: str = "mixed") -> Optional[Dict[str, Any]]:

        if not image_urls:
            return None
        
        print(f"\n[ImageSelectorAgent] 评估 {len(image_urls)} 张候选图片...")
        
        if len(image_urls) == 1:
            print(f"[ImageSelectorAgent] 只有一张图片，直接评估...")
            can_use, result_dict = self._evaluate_single_image_by_url(
                image_urls[0], plan_summary, question_type, edit_instruction, image_type
            )
            
            reasoning = result_dict.get("reasoning", "No reasoning") if isinstance(result_dict, dict) else str(result_dict)
            
            image_pil = self._download_or_load_image(image_urls[0])
            if not image_pil:
                print(f"[ImageSelectorAgent] 无法加载图片")
                return None
            
            return {
                "selected_url": image_urls[0],
                "selected_image": image_pil,
                "can_use_directly": can_use,
                "reasoning": reasoning
            }
        
        print(f"[ImageSelectorAgent] 逐张评估...")
        evaluations = []
        for idx, url in enumerate(image_urls):
            print(f"\n  评估图片 {idx+1}/{len(image_urls)}: {url}...")
            can_use, result_dict = self._evaluate_single_image_by_url(
                url, plan_summary, question_type, edit_instruction, image_type
            )
            score = self._extract_score_from_reasoning(result_dict)
            reasoning = result_dict.get("reasoning", "No reasoning") if isinstance(result_dict, dict) else str(result_dict)
            
            evaluations.append({
                "index": idx,
                "url": url,
                "can_use": can_use,
                "reasoning": reasoning,
                "score": score
            })
            print(f"    评分: {score}/10, 可直接使用: {can_use}")
            print(f"    理由: {reasoning}...")
        
        print(f"\n[ImageSelectorAgent] 智能选择策略：")
        print(f"  优先级1: 非原图 + 可直接使用 ✨")
        print(f"  优先级2: 任何非原图（需要生成）")
        
        original_url = f"file://{original_image_path}" if original_image_path else None
        
        non_original_usable = [
            e for e in evaluations 
            if e["can_use"] and e["url"] != original_url
        ]
        
        if non_original_usable:
            best = max(non_original_usable, key=lambda x: x["score"])
            print(f"  ✓ 找到可直接使用的搜索图片: 图片{best['index']+1} (评分:{best['score']}/10)")
            print(f"  → 将直接基于此图提问，跳过生成 ⚡")
        else:
            non_original_all = [e for e in evaluations if e["url"] != original_url]
            
            if non_original_all:
                best = max(non_original_all, key=lambda x: x["score"])
                print(f"  ✗ 无可直接使用的搜索图片")
                print(f"  → 选择最佳非原图: 图片{best['index']+1} (评分:{best['score']}/10)")
                print(f"  → 将触发图片编辑/生成 🎨")
                best["can_use"] = False  
            else:
                best = max(evaluations, key=lambda x: x["score"])
                print(f"  ⚠️  警告: 所有候选都是原图！")
                print(f"  → 强制触发图片编辑/生成")
                best["can_use"] = False  
        
        selected_image = self._download_or_load_image(best["url"])
        if not selected_image:
            print(f"[ImageSelectorAgent] ⚠️  无法加载选中图片，尝试其他候选...")
            fallback_candidates = sorted(
                [e for e in evaluations if e["url"] != original_url and e["index"] != best["index"]],
                key=lambda x: x["score"],
                reverse=True
            )
            
            for eval_item in fallback_candidates:
                selected_image = self._download_or_load_image(eval_item["url"])
                if selected_image:
                    best = eval_item
                    if eval_item["url"] == original_url or not eval_item["can_use"]:
                        best["can_use"] = False
                    print(f"[ImageSelectorAgent] 改用备选图片 {best['index']+1}")
                    break
            
            if not selected_image:
                print(f"[ImageSelectorAgent] ❌ 所有图片都无法加载")
                return None
        
        return {
            "selected_url": best["url"],
            "selected_image": selected_image,
            "can_use_directly": best["can_use"],
            "reasoning": best["reasoning"]
        }
    
    def _download_or_load_image(self, url: str) -> Optional[Image.Image]:

        try:
            if url.startswith("http://") or url.startswith("https://"):
                return self.image_edit_client.download_image(url)
            else:
                local_path = url.replace("file://", "")
                return Image.open(local_path)
        except Exception as e:
            print(f"[ImageSelectorAgent] 加载图片失败 {url}: {e}")
            return None
    
    def _evaluate_single_image_by_url(self, image_url: str, plan_summary: str,
                                     question_type: str, edit_instruction: str = "",
                                     image_type: str = "mixed") -> Tuple[bool, Dict[str, Any]]:

        edit_note = ""
        if edit_instruction:
            edit_note = f"\n\nNote: The plan includes this edit instruction: \"{edit_instruction}\"\nIf the image does NOT already contain these modifications, set can_use_directly to FALSE."
        
        try:
            img_type_enum = ImageType(image_type)
            type_specific_guidance = ImageTypeStrategy.get_evaluation_prompt_suffix(img_type_enum)
            threshold = ImageTypeStrategy.get_can_use_threshold(img_type_enum)
        except ValueError:
            type_specific_guidance = ""
            threshold = 7.0
        
        prompt = f"""You are an Image Quality Evaluator. Evaluate if this image can be used to generate a MEANINGFUL and HIGH-QUALITY {question_type} question.

Plan Summary: {plan_summary}
Question Type: {question_type}
Image Type: {image_type.upper()}{edit_note}{type_specific_guidance}

EVALUATION CRITERIA (Be lenient - prioritize using existing images):

✅ SET can_use_directly = TRUE if the image:
   - Contains sufficient visual information for asking a {question_type} question
   - Has reasonable clarity (doesn't need to be perfect)
   - Relates to the plan goal (even loosely)
   - Could lead to an interesting, answerable question
   
❌ SET can_use_directly = FALSE ONLY if:
   - The image is completely unrelated to the plan
   - The image is severely blurred/corrupted/unusable
   - Critical information is missing that makes questions impossible
   
IMPORTANT: Prefer TRUE over FALSE. We want to USE searched images whenever reasonable, not generate new ones unnecessarily.

Your task:
1. Examine the image content
2. Score its quality and relevance (1-10)
3. Decide: Can we ask a good {question_type} question based on this image?

Output format (JSON):
{{
    "can_use_directly": true/false,
    "score": 7,
    "reasoning": "Brief explanation (what question aspects this image supports)"
}}

Output ONLY the JSON, no extra text."""

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    print(f"[ImageSelectorAgent] 等待 {wait_time}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                
                messages = [{"role": "user", "content": prompt}]
                
                if image_url.startswith("http://") or image_url.startswith("https://"):
                    response = self.client.chat(messages, image_url=image_url, max_tokens=8192)
                else:
                    local_path = image_url.replace("file://", "")
                    response = self.client.chat(messages, image_path=local_path, max_tokens=8192)
                
                print(f"[ImageSelectorAgent API Response]:\n{response}\n")
                
                if not response:
                    print(f"[ImageSelectorAgent] API返回None，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    print(f"[ImageSelectorAgent] 错误: 响应中未找到JSON格式")
                    print(f"  - 原始响应: {response}")
                    continue
                
                json_str = json_match.group()
                json_str = re.sub(r',(\s*)}', r'\1}', json_str)  
                json_str = re.sub(r',(\s*)]', r'\1]', json_str)  
                
                result = json.loads(json_str)
                
                can_use = result.get("can_use_directly", False)
                if not isinstance(can_use, bool):
                    print(f"[ImageSelectorAgent] 警告: can_use_directly字段类型错误，默认为False")
                    print(f"  - 实际类型: {type(can_use)}, 值: {can_use}")
                    can_use = False
                
                if "score" not in result:
                    result["score"] = 5
                
                if attempt > 0:
                    print(f"[ImageSelectorAgent] ✓ 重试成功（尝试 {attempt + 1}/{MAX_RETRIES}）")
                return can_use, result
                
            except json.JSONDecodeError as e:
                print(f"[ImageSelectorAgent] JSON解析失败，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 错误: {e}")
                print(f"  - 尝试解析的内容: {response if 'response' in locals() else 'N/A'}")
                continue
            except Exception as e:
                print(f"[ImageSelectorAgent] 异常，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 异常类型: {type(e).__name__}")
                print(f"  - 错误信息: {str(e)}")
                continue
        
        print(f"[ImageSelectorAgent] ❌ 所有重试失败，已尝试 {MAX_RETRIES} 次")
        return False, {"can_use_directly": False, "score": 0, "reasoning": "Evaluation failed: 所有重试均失败，请查看上方日志获取详细信息"}
    
    def _extract_score_from_reasoning(self, result: Dict[str, Any]) -> float:
        if isinstance(result, dict) and "score" in result:
            try:
                score = float(result["score"])
                return score
            except:
                pass
        
        if isinstance(result, str):
            try:
                import json
                parsed = json.loads(result)
                if "score" in parsed:
                    return float(parsed["score"])
            except:
                pass
        
        reasoning = result.get("reasoning", "") if isinstance(result, dict) else str(result)
        if "perfect" in reasoning.lower() or "excellent" in reasoning.lower():
            return 9.0
        elif "good" in reasoning.lower():
            return 7.0
        elif "acceptable" in reasoning.lower() or "okay" in reasoning.lower():
            return 5.0
        else:
            return 3.0


class QuestionGeneratorAgent:
    
    def __init__(self, weakness_context: str = ""):
        self.client = O3Client(model_name=O3_MODEL)
        self.use_two_stage = True  # 启用两阶段生成
        self.weakness_context = weakness_context
    
    def _observe_image(self, image_pil: Image.Image) -> Optional[str]:
        
        prompt = """You are a Visual Observer. Describe what you see in this image in detail.

🔍 OBSERVATION TASK:
Provide a comprehensive, objective description of the image content:

1. **Main Subject**: What is the primary focus of the image?
2. **Key Elements**: What objects, people, text, numbers, or structures are visible?
3. **Spatial Layout**: How are elements arranged or positioned?
4. **Quantities**: Are there countable items? How many?
5. **Text Content**: Any visible text, labels, numbers, or symbols?
6. **Visual Characteristics**: Colors, shapes, patterns, relationships?
7. **Context**: What type of image is this (map, diagram, photo, chart, etc.)?

❗ IMPORTANT:
- Be OBJECTIVE and FACTUAL - only describe what you actually see
- Do NOT make assumptions or interpretations beyond visual content
- Do NOT suggest questions or analysis - just observe and describe
- Include ALL visible details that might be useful for asking questions

Output your observation in clear, structured text (not JSON)."""

        for attempt in range(3): 
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat(messages, image_pil=image_pil, max_tokens=2048)
                
                if response and len(response.strip()) > 50:
                    print(f"\n[QuestionGeneratorAgent] 📸 图片观察结果:")
                    print(f"{'-'*80}")
                    print(response[:500] + ("..." if len(response) > 500 else ""))
                    print(f"{'-'*80}\n")
                    return response
                else:
                    print(f"[QuestionGeneratorAgent] 观察结果太短或为空，重试 {attempt + 1}/3")
                    continue
                    
            except Exception as e:
                print(f"[QuestionGeneratorAgent] 观察阶段异常 {attempt + 1}/3: {e}")
                continue
        
        print("[QuestionGeneratorAgent] ⚠️  图片观察失败，将使用单阶段生成")
        return None

    def _generate_from_observation(self, observation: str, question_type: str, 
                                   plan_summary: str = "") -> Optional[Dict[str, str]]:

        context = ""
        if plan_summary:
            context = f"\nGeneral Goal: {plan_summary}\n"
        
        weakness_info = ""
        if self.weakness_context:
            weakness_info = f"\n\n🎯 FOCUS AREAS (based on identified weaknesses):\n{self.weakness_context}\n\nPlease prioritize generating questions that target these specific areas.\n"
        
        prompt = f"""You are a Visual Question Generator. Based on the following observation of an image, generate ONE high-quality question.

📸 IMAGE OBSERVATION:
{observation}

🎯 YOUR TASK:
Generate a {question_type} question based EXCLUSIVELY on the observation above.
{context}{weakness_info}

📋 REQUIREMENTS:

1. **Content Alignment**: 
   - Your question MUST ask about things mentioned in the observation
   - Do NOT introduce new elements not described in the observation
   - Every part of your question should be verifiable from the observation

2. **Question Type**: {question_type}
   • If "multiple choice": MUST provide ALL 4 options in the question text
     FORMAT EXAMPLE:
     {{
         "question": "What color is the sky?\\nA. Red\\nB. Blue\\nC. Green\\nD. Yellow",
         "answer": "B"
     }}
     CRITICAL: Include "A. ... B. ... C. ... D. ..." directly in the question field
     - Options must be based on observable details
     - Correct answer must be one of A, B, C, or D
   • If "numerical": Ask for a specific count, measurement, or calculation
     - Must be answerable with a precise number from the observation
   • If "regression": Ask for a continuous value prediction
     - Provide reasoning based on observable patterns

3. **Quality Standards**:
   ✅ Clear and unambiguous
   ✅ Requires observation or reasoning (not just memorization)
   ✅ Has ONE correct answer determinable from the image
   ✅ Appropriate difficulty (not too trivial, not impossible)
   
   ❌ AVOID:
   - Vague questions ("what about...", "describe...")
   - Questions requiring external knowledge
   - Questions about unobserved details

📤 OUTPUT FORMAT (JSON only, no extra text):
{{
    "question": "Your question based on the observation",
    "answer": "The correct answer"
}}

💡 TIP: First identify the most interesting/questionable aspect from the observation, then formulate your question around that."""

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    print(f"[QuestionGeneratorAgent] 等待 {wait_time}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat(messages, max_tokens=8192)  # 不传图片，只基于观察
                
                print(f"[QuestionGeneratorAgent API Response]:\n{response}\n")
                
                if not response:
                    print(f"[QuestionGeneratorAgent] API返回None，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    print(f"[QuestionGeneratorAgent] 错误: 响应中未找到JSON格式")
                    print(f"  - 原始响应: {response}")
                    continue
                
                json_str = json_match.group()
                json_str = re.sub(r',(\s*)}', r'\1}', json_str)  # 移除 },} 中的逗号
                json_str = re.sub(r',(\s*)]', r'\1]', json_str)  # 移除 ],] 中的逗号
                
                result = json.loads(json_str)
                
                if "question" not in result or not result["question"]:
                    print(f"[QuestionGeneratorAgent] 错误: 缺少question字段，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                if "answer" not in result or not result["answer"]:
                    print(f"[QuestionGeneratorAgent] 错误: 缺少answer字段，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                if question_type == "multiple choice":
                    question_text = result["question"]

                    has_options = bool(
                        re.search(r'[A-D]\.\s*.+', question_text) and
                        len(re.findall(r'[A-D]\.', question_text)) >= 3
                    )
                    
                    if not has_options:
                        print(f"[QuestionGeneratorAgent] ❌ 多选题格式错误: 缺少A/B/C/D选项")
                        print(f"  问题内容: {question_text[:200]}...")
                        print(f"  要求: 必须在问题中包含 'A. ... B. ... C. ... D. ...' 格式")
                        print(f"  尝试 {attempt + 1}/{MAX_RETRIES}，重新生成...\n")
                        continue
                    
                    answer = result["answer"].strip()
                    if not re.match(r'^[A-D]\.?', answer, re.IGNORECASE):
                        print(f"[QuestionGeneratorAgent] ⚠️  多选题答案格式不规范: '{answer}'")
                        print(f"  自动规范化为选项字母...")
                        letter_match = re.search(r'[A-D]', answer, re.IGNORECASE)
                        if letter_match:
                            result["answer"] = letter_match.group().upper()
                            print(f"  规范化后: {result['answer']}")
                        else:
                            print(f"  ❌ 无法提取有效选项，重新生成...")
                            continue
                
                print(f"[QuestionGeneratorAgent] ✓ 字段验证通过")
                if question_type == "multiple choice":
                    print(f"  ✓ 多选题格式验证通过 (包含选项)")
                return result
                
            except json.JSONDecodeError as e:
                print(f"[QuestionGeneratorAgent] JSON解析失败，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 错误: {e}")
                continue
            except Exception as e:
                print(f"[QuestionGeneratorAgent] 异常，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 异常类型: {type(e).__name__}")
                print(f"  - 错误信息: {str(e)}")
                continue
        
        return None

    def generate_question(self, image_pil: Image.Image, question_type: str,
                         original_question: str = "", original_answer: str = "",
                         plan_summary: str = "") -> Optional[Dict[str, str]]:

        print(f"\n{'='*80}")
        print(f"[QuestionGeneratorAgent] 🎯 开始生成问题")
        print(f"  问题类型: {question_type}")
        print(f"  生成策略: {'两阶段（观察→提问）' if self.use_two_stage else '单阶段'}")
        print(f"{'='*80}")
        
        if self.use_two_stage:
            print("\n[QuestionGeneratorAgent] 📝 阶段1: 观察图片...")
            observation = self._observe_image(image_pil)
            
            if observation:
                print("\n[QuestionGeneratorAgent] 📝 阶段2: 基于观察生成问题...")
                result = self._generate_from_observation(observation, question_type, plan_summary)
                
                if result:
                    print(f"\n[QuestionGeneratorAgent] ✅ 两阶段生成成功")
                    print(f"  问题: {result['question'][:100]}...")
                    print(f"  答案: {result['answer'][:50]}...")
                    print(f"{'='*80}\n")
                    return result
                else:
                    print("\n[QuestionGeneratorAgent] ⚠️  阶段2失败，回退到单阶段模式")
            else:
                print("\n[QuestionGeneratorAgent] ⚠️  阶段1失败，回退到单阶段模式")
        
        print("\n[QuestionGeneratorAgent] 📝 使用单阶段生成（备用方案）...")
        return self._single_stage_generate(image_pil, question_type, original_question, 
                                          original_answer, plan_summary)
    
    def _single_stage_generate(self, image_pil: Image.Image, question_type: str,
                               original_question: str, original_answer: str,
                               plan_summary: str) -> Optional[Dict[str, str]]:

        context_parts = []
        
        if plan_summary:
            context_parts.append(f"Goal: {plan_summary}")
        
        if original_question:
            context_parts.append(f"""
⚠️ REFERENCE (from a DIFFERENT image - do NOT copy):
- Previous question: "{original_question}"
- DO NOT ask about things in that question if they are NOT in THIS image
- Generate based ONLY on what you SEE in the provided image""")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        format_note = ""
        if question_type == "multiple choice":
            format_note = """
⚠️ CRITICAL for MULTIPLE CHOICE:
- Include ALL 4 options (A, B, C, D) in the question field
- Format: "Your question?\\nA. Option1\\nB. Option2\\nC. Option3\\nD. Option4"
- Answer should be just the letter (A, B, C, or D)
"""
        
        prompt = f"""You are a Visual Question Generator. Generate ONE question based EXCLUSIVELY on what you see in the image.

Question Type: {question_type}
{context}
{format_note}

🔍 PROCESS:
1. OBSERVE the image carefully
2. IDENTIFY interesting aspects
3. GENERATE a question based ONLY on visual content

Output JSON:
{{
    "question": "Your question{' with A, B, C, D options' if question_type == 'multiple choice' else ''}",
    "answer": "The answer{' (just the letter)' if question_type == 'multiple choice' else ''}"
}}"""

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    time.sleep(wait_time)
                
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat(messages, image_pil=image_pil, max_tokens=8192)
                
                if not response:
                    continue
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    continue
                
                json_str = json_match.group()
                json_str = re.sub(r',(\s*)}', r'\1}', json_str)  # 移除 },} 中的逗号
                json_str = re.sub(r',(\s*)]', r'\1]', json_str)  # 移除 ],] 中的逗号
                
                result = json.loads(json_str)
                
                if "question" in result and result["question"] and \
                   "answer" in result and result["answer"]:
                    
                    if question_type == "multiple choice":
                        question_text = result["question"]
                        has_options = bool(
                            re.search(r'[A-D]\.\s*.+', question_text) and
                            len(re.findall(r'[A-D]\.', question_text)) >= 3
                        )
                        
                        if not has_options:
                            print(f"[QuestionGeneratorAgent] ❌ 单阶段多选题缺少选项，重试...")
                            continue
                        
                        answer = result["answer"].strip()
                        if not re.match(r'^[A-D]\.?$', answer, re.IGNORECASE):
                            letter_match = re.search(r'[A-D]', answer, re.IGNORECASE)
                            if letter_match:
                                result["answer"] = letter_match.group().upper()
                            else:
                                print(f"[QuestionGeneratorAgent] ❌ 单阶段答案格式错误，重试...")
                                continue
                    
                    print(f"[QuestionGeneratorAgent] ✓ 单阶段生成成功")
                    return result
                    
            except Exception as e:
                print(f"[QuestionGeneratorAgent] 单阶段异常 {attempt + 1}/{MAX_RETRIES}: {e}")
                continue
        
        print(f"[QuestionGeneratorAgent] ❌ 所有尝试失败")
        return None
    
    def generate_divergent_question(self, image_pil: Image.Image, question_type: str,
                                   source_question: str, source_answer: str,
                                   plan_summary: str = "") -> Optional[Dict[str, str]]:

        print(f"\n{'='*80}")
        print(f"[QuestionGeneratorAgent] 🔄 发散生成模式")
        print(f"  问题类型: {question_type}")
        print(f"  源问题: {source_question[:100]}...")
        print(f"{'='*80}")
        
        print("\n[QuestionGeneratorAgent] 📝 阶段1: 观察图片...")
        observation = self._observe_image(image_pil)
        
        if not observation:
            print("[QuestionGeneratorAgent] ⚠️  图片观察失败，无法进行发散生成")
            return None
        
        print("\n[QuestionGeneratorAgent] 📝 阶段2: 发散生成新问题...")
        
        context = ""
        if plan_summary:
            context = f"\nGeneral Goal: {plan_summary}\n"
        
        weakness_info = ""
        if self.weakness_context:
            weakness_info = f"\n\n🎯 FOCUS AREAS (based on identified weaknesses):\n{self.weakness_context}\n\nPlease prioritize generating questions that target these specific areas.\n"
        
        prompt = f"""You are a Visual Question Generator in DIVERGENT MODE. Your task is to generate a NEW question that is:
1. Based on the same image (described below)
2. Similar in TYPE and STYLE to the reference question
3. But asks about DIFFERENT aspects of the image

📸 IMAGE OBSERVATION:
{observation}

📚 REFERENCE QUESTION (for inspiration only - DO NOT copy):
Question: {source_question}
Answer: {source_answer}
Type: {question_type}

🎯 YOUR TASK:
Generate ONE NEW {question_type} question that:
- Is based on the SAME image (use the observation above)
- Has the SAME question type and style as the reference
- But focuses on DIFFERENT visual elements or aspects
- Is of similar difficulty level
{context}{weakness_info}

📋 REQUIREMENTS:

1. **Divergent Thinking**:
   - If reference asks about colors, you could ask about shapes, quantities, positions, etc.
   - If reference asks about the left side, consider asking about the right side or center
   - If reference asks about one object, consider asking about another object
   - Explore DIFFERENT angles and perspectives on the same image

2. **Content Alignment**: 
   - Your question MUST be answerable from the observation
   - Do NOT introduce elements not mentioned in the observation
   - Stay grounded in what's actually visible

3. **Question Type**: {question_type}
   • If "multiple choice": MUST provide ALL 4 options in the question text
     FORMAT EXAMPLE:
     {{
         "question": "What shape is the main object?\\nA. Circle\\nB. Square\\nC. Triangle\\nD. Rectangle",
         "answer": "B"
     }}
     - Include "A. ... B. ... C. ... D. ..." directly in the question field
     - Options must be based on observable details
     - Correct answer must be one of A, B, C, or D
   • If "numerical": Ask for a specific count or measurement
   • If "regression": Ask for a continuous value with reasoning

4. **Quality Standards**:
   ✅ Clear and unambiguous
   ✅ Different focus from the reference question
   ✅ Requires observation or reasoning
   ✅ Has ONE correct answer determinable from the image
   ✅ Appropriate difficulty level
   
   ❌ AVOID:
   - Copying or slightly modifying the reference question
   - Vague questions
   - Questions requiring external knowledge
   - Questions about unobserved details

📤 OUTPUT FORMAT (JSON only, no extra text):
{{
    "question": "Your NEW divergent question",
    "answer": "The correct answer"
}}

💡 STRATEGY:
1. Review the observation to find aspects NOT covered by the reference question
2. Identify interesting visual elements or patterns
3. Formulate a question that explores these different aspects
4. Ensure it's answerable and has similar style to the reference"""

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    print(f"[QuestionGeneratorAgent] 等待 {wait_time}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat(messages, max_tokens=8192)
                
                print(f"[QuestionGeneratorAgent Divergent API Response]:\n{response}\n")
                
                if not response:
                    print(f"[QuestionGeneratorAgent] API返回None，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    print(f"[QuestionGeneratorAgent] 错误: 响应中未找到JSON格式")
                    continue
                
                json_str = json_match.group()
                json_str = re.sub(r',(\s*)}', r'\1}', json_str)
                json_str = re.sub(r',(\s*)]', r'\1]', json_str)
                
                result = json.loads(json_str)
                
                if "question" not in result or not result["question"]:
                    print(f"[QuestionGeneratorAgent] 错误: 缺少question字段")
                    continue
                
                if "answer" not in result or not result["answer"]:
                    print(f"[QuestionGeneratorAgent] 错误: 缺少answer字段")
                    continue
                
                if question_type == "multiple choice":
                    question_text = result["question"]
                    has_options = bool(
                        re.search(r'[A-D]\.\s*.+', question_text) and
                        len(re.findall(r'[A-D]\.', question_text)) >= 3
                    )
                    
                    if not has_options:
                        print(f"[QuestionGeneratorAgent] ❌ 多选题格式错误: 缺少选项")
                        continue
                    
                    answer = result["answer"].strip()
                    if not re.match(r'^[A-D]\.?', answer, re.IGNORECASE):
                        letter_match = re.search(r'[A-D]', answer, re.IGNORECASE)
                        if letter_match:
                            result["answer"] = letter_match.group().upper()
                        else:
                            continue
                
                print(f"[QuestionGeneratorAgent] ✅ 发散生成成功")
                print(f"  新问题: {result['question'][:100]}...")
                print(f"  答案: {result['answer']}")
                return result
                
            except json.JSONDecodeError as e:
                print(f"[QuestionGeneratorAgent] JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"[QuestionGeneratorAgent] 异常: {type(e).__name__} - {str(e)}")
                continue
        
        print(f"[QuestionGeneratorAgent] ❌ 发散生成失败，已尝试 {MAX_RETRIES} 次")
        return None


class ValidationAgent:
    
    def __init__(self):
        self.client = O3Client(model_name=CLAUDE_MODEL)
    
    def validate(self, question: str, answer: str, image_pil: Image.Image,
                question_type: str) -> Tuple[bool, str]:

        if question_type == "multiple choice":
            has_options = bool(
                re.search(r'[A-D]\.\s*.+', question) and
                len(re.findall(r'[A-D]\.', question)) >= 3
            )
            
            if not has_options:
                print(f"[ValidationAgent] ❌ 格式预检失败: 多选题缺少A/B/C/D选项")
                print(f"  问题: {question[:200]}...")
                return False, "FAILED: Multiple choice question must include options (A, B, C, D) in the question text"
            
            if not re.search(r'[A-D]', answer, re.IGNORECASE):
                print(f"[ValidationAgent] ❌ 格式预检失败: 多选题答案必须是A/B/C/D之一")
                print(f"  答案: {answer}")
                return False, "FAILED: Multiple choice answer must be one of A, B, C, or D"
            
            print(f"[ValidationAgent] ✓ 多选题格式预检通过")
        
        prompt = f"""You are a Visual Question Quality Validator. Your task is to verify if a generated question is appropriate for the provided image.

📋 QUESTION TO VALIDATE:
Question: {question}
Answer: {answer}
Expected Type: {question_type}

🔍 VALIDATION PROCESS:

STEP 1: EXAMINE THE IMAGE
- What do you actually see in the image?
- What information is visually available?

STEP 2: CHECK QUESTION-IMAGE ALIGNMENT ⭐ MOST CRITICAL
✅ PASS if:
- The question asks about things VISIBLE in the image
- The answer can be DIRECTLY determined by looking at the image
- All mentioned objects/concepts/numbers are PRESENT in the image

❌ FAIL if:
- The question asks about things NOT in the image
- The question seems based on assumptions rather than visual content
- The answer requires information not visible in the image
- The question mentions specific items that are not actually present

STEP 3: CHECK TECHNICAL REQUIREMENTS
✅ PASS if:
- Question is clear and unambiguous
- Question type matches: {question_type}
  • "multiple choice": MUST have ALL 4 options (A, B, C, D) in the question text
    Example: "What color?\\nA. Red\\nB. Blue\\nC. Green\\nD. Yellow"
    Answer must be one letter (A, B, C, or D)
  • "numerical": Requires a specific number answer
  • "regression": Requires a continuous value prediction
- Answer is specific and correct
- Question requires observation/reasoning (not just description)

⚠️ PRIORITY: Question-Image alignment is MORE IMPORTANT than complexity or difficulty.
A simple but accurate question is BETTER than a complex question about invisible things.

Output MUST be in valid JSON format:
{{
    "is_valid": true/false,
    "reasoning": "Specific explanation: What did you see in the image? Does the question match what's visible?"
}}

Output ONLY the JSON, no extra text."""

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt * BASE_DELAY, MAX_DELAY)
                    print(f"[ValidationAgent] 等待 {wait_time}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat(messages, image_pil=image_pil, max_tokens=8192)
                
                print(f"[ValidationAgent API Response]:\n{response}\n")
                
                if not response:
                    print(f"[ValidationAgent] API返回None，尝试 {attempt + 1}/{MAX_RETRIES}")
                    continue
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    print(f"[ValidationAgent] 错误: 响应中未找到JSON格式")
                    print(f"  - 原始响应: {response}")
                    continue
                
                json_str = json_match.group()
                json_str = re.sub(r',(\s*)}', r'\1}', json_str)  
                json_str = re.sub(r',(\s*)]', r'\1]', json_str)  
                
                result = json.loads(json_str)
                
                is_valid = result.get("is_valid", False)
                if not isinstance(is_valid, bool):
                    print(f"[ValidationAgent] 警告: is_valid字段类型错误，默认为False")
                    print(f"  - 实际类型: {type(is_valid)}, 值: {is_valid}")
                    is_valid = False
                
                reasoning = result.get("reasoning", "No reasoning provided")
                if not reasoning or not isinstance(reasoning, str):
                    print(f"[ValidationAgent] 警告: reasoning字段无效")
                    reasoning = "No valid reasoning provided"
                
                print(f"[ValidationAgent] ✓ 验证完成")
                if attempt > 0:
                    print(f"[ValidationAgent] ✓ 重试成功（尝试 {attempt + 1}/{MAX_RETRIES}）")
                return is_valid, reasoning
                
            except json.JSONDecodeError as e:
                print(f"[ValidationAgent] JSON解析失败，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 错误: {e}")
                print(f"  - 尝试解析的内容: {response if 'response' in locals() else 'N/A'}")
                continue
            except Exception as e:
                print(f"[ValidationAgent] 异常，尝试 {attempt + 1}/{MAX_RETRIES}")
                print(f"  - 异常类型: {type(e).__name__}")
                print(f"  - 错误信息: {str(e)}")
                continue
        
        print(f"[ValidationAgent] ❌ 所有重试失败，已尝试 {MAX_RETRIES} 次")
        return False, "Validation failed: 所有重试均失败，请查看上方日志获取详细信息"


