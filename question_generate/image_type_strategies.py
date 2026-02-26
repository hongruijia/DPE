from typing import Dict, Any, List
from enum import Enum
import os
import requests
import json
from functools import lru_cache


GPT4O_MODEL = "gpt-4o-2024-05-13"
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
REQUEST_TIMEOUT = 60


class ImageType(Enum):
    """图片类型分类"""
    GEOMETRIC = "geometric"           
    MEDICAL = "medical"             
    CHART_GRAPH = "chart_graph"     
    TEXT_DENSE = "text_dense"    
    DIAGRAM = "diagram"      
    MATH_FORMULA = "math_formula"    
    MAP = "map"          
    NATURAL_SCENE = "natural_scene"   
    ARTISTIC = "artistic"   
    EVERYDAY_OBJECT = "everyday_object" 
    ARCHITECTURAL = "architectural"   
    MIXED = "mixed"       


@lru_cache(maxsize=1000)
def identify_image_type_with_gpt4o(plan_summary: str, question_type: str, original_question: str = "") -> str:

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{DASHSCOPE_API_KEY}"
    }
    
    user_content = f"""Based on the following information, identify the IMAGE TYPE of the question.

Plan Summary: {plan_summary}
Question Type: {question_type}
Original Question: {original_question}

Available IMAGE TYPES (choose ONE that best fits):
1. geometric - Geometric shapes, angles, triangles, circles, polygons
2. medical - Medical images, X-rays, CT scans, anatomical diagrams
3. chart_graph - Charts, graphs, data visualizations, statistics
4. text_dense - Text-heavy documents, receipts, menus, OCR content
5. diagram - Flowcharts, schematics, circuit diagrams, process diagrams
6. math_formula - Mathematical formulas, equations, expressions
7. map - Maps, geographic locations, routes, navigation
8. natural_scene - Natural landscapes, outdoor scenes, nature
9. artistic - Artistic images, paintings, creative designs
10. everyday_object - Common objects, products, furniture, household items
11. architectural - Buildings, structures, architectural designs
12. mixed - Mixed or unclear category

Please respond with ONLY ONE word from the list above (e.g., "geometric", "chart_graph", etc.). Do not include any explanation or additional text."""
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert at classifying image types based on question content. Always respond with exactly one image type category from the provided list."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    payload = {
        "model": GPT4O_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 20
    }
    
    max_retries = 3
    for retry in range(max_retries):
        try:
            response = requests.post(
                DASHSCOPE_BASE_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                if retry < max_retries - 1:
                    print(f"[identify_image_type_with_gpt4o] HTTP错误: {response.status_code}, 重试 {retry + 1}/{max_retries}")
                    continue
                else:
                    print(f"[identify_image_type_with_gpt4o] HTTP错误: {response.status_code}, 使用备用方法")
                    return "mixed"
            
            result_json = response.json()
            
            if 'choices' in result_json and len(result_json['choices']) > 0:
                content = result_json['choices'][0].get('message', {}).get('content', '').strip().lower()
                
                valid_types = [
                    "geometric", "medical", "chart_graph", "text_dense", "diagram",
                    "math_formula", "map", "natural_scene", "artistic", "everyday_object",
                    "architectural", "mixed"
                ]
                
                first_word = content.split()[0] if content.split() else "mixed"
                
                if first_word in valid_types:
                    return first_word
                else:
                    for valid_type in valid_types:
                        if valid_type in content:
                            return valid_type
                    
                    print(f"[identify_image_type_with_gpt4o] 无效的类型: {content}, 使用 mixed")
                    return "mixed"
            else:
                if retry < max_retries - 1:
                    print(f"[identify_image_type_with_gpt4o] 响应格式异常, 重试 {retry + 1}/{max_retries}")
                    continue
                else:
                    print(f"[identify_image_type_with_gpt4o] 响应格式异常: {result_json}")
                    return "mixed"
                
        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                print(f"[identify_image_type_with_gpt4o] 请求超时, 重试 {retry + 1}/{max_retries}")
                continue
            else:
                print(f"[identify_image_type_with_gpt4o] 请求超时，使用备用方法")
                return "mixed"
        except Exception as e:
            if retry < max_retries - 1:
                print(f"[identify_image_type_with_gpt4o] 错误: {e}, 重试 {retry + 1}/{max_retries}")
                continue
            else:
                print(f"[identify_image_type_with_gpt4o] 错误: {e}，使用备用方法")
                return "mixed"
    
    return "mixed"


class ImageTypeStrategy:
    
    STRATEGIES = {
        ImageType.GEOMETRIC: {
            "priority": "use_search_heavily",  
            "edit_allowed": "minimal",   
            "edit_scope": "tiny_adjustments",  
            "quality_threshold": 9.0,  
            "can_use_threshold": 7.5,  
            "edit_instruction_template": """
CRITICAL: This is a GEOMETRIC image requiring EXTREME PRECISION.

Allowed edits:
- Add/modify LABELS or ANNOTATIONS only (e.g., angle labels, length markers)
- Adjust COLORS for clarity (e.g., highlight specific angles)
- Add MEASUREMENT indicators

FORBIDDEN:
- DO NOT change shapes, angles, proportions
- DO NOT distort or transform geometric figures
- DO NOT add/remove geometric elements

Edit instructions: {instruction}

Remember: Preserve ALL geometric properties exactly.
""",
            "description": "几何图形 - 要求极高精确性，优先搜索，仅允许标注和颜色调整"
        },
        
        ImageType.MEDICAL: {
            "priority": "use_search_heavily",
            "edit_allowed": "minimal",
            "edit_scope": "annotations_only",  
            "quality_threshold": 9.5,
            "can_use_threshold": 7.0,
            "edit_instruction_template": """
CRITICAL: This is a MEDICAL image requiring EXTREME ACCURACY.

Allowed edits:
- Add ANNOTATIONS (arrows, circles, labels pointing to anatomical structures)
- Add TEXT labels for medical terminology
- Adjust CONTRAST or BRIGHTNESS for clarity

STRICTLY FORBIDDEN:
- DO NOT alter anatomical structures
- DO NOT modify tissue appearances
- DO NOT change medical imaging characteristics

Edit instructions: {instruction}

Remember: Medical accuracy is paramount. Any structural change is unacceptable.
""",
            "description": "医学影像 - 要求极高准确性，优先搜索，仅允许注释和对比度调整"
        },
        
        ImageType.CHART_GRAPH: {
            "priority": "use_search_preferred",  
            "edit_allowed": "moderate",      
            "edit_scope": "data_labels",      
            "quality_threshold": 8.5,
            "can_use_threshold": 7.0,
            "edit_instruction_template": """
This is a CHART/GRAPH requiring data accuracy.

Allowed edits:
- Modify AXIS LABELS or LEGEND text
- Change BAR/LINE COLORS for visual distinction
- Add/modify DATA POINT LABELS (numbers)
- Adjust TITLE or CAPTION

Important constraints:
- Keep data PROPORTIONS accurate if changing values
- Maintain SCALE consistency
- Preserve TREND patterns

Edit instructions: {instruction}
""",
            "description": "图表数据 - 保持数据比例，可修改标签和颜色"
        },
        
        ImageType.TEXT_DENSE: {
            "priority": "use_search_preferred",
            "edit_allowed": "targeted",         
            "edit_scope": "key_text_only",     
            "quality_threshold": 8.0,
            "can_use_threshold": 7.0,
            "edit_instruction_template": """
This is a TEXT-DENSE image. Focus on specific text modifications.

Editing strategy:
1. IDENTIFY the key text elements to modify: {instruction}
2. PRESERVE the overall document layout and structure
3. Modify ONLY the specified text portions
4. Keep FONT STYLE, SIZE consistent with surrounding text

Example approach:
- If modifying a title: Change only the title text, keep formatting
- If modifying a number: Replace specific digits, keep context
- If modifying a word: Replace in-place, maintain text flow

AVOID:
- Changing large blocks of text (OCR may fail)
- Modifying decorative elements or backgrounds
- Altering document structure

Specific changes needed: {instruction}
""",
            "description": "文字密集 - 仅修改指定关键文字，保持布局"
        },
        
        ImageType.DIAGRAM: {
            "priority": "use_search_preferred",
            "edit_allowed": "moderate",
            "edit_scope": "structure_aware", 
            "quality_threshold": 8.0,
            "can_use_threshold": 7.0,
            "edit_instruction_template": """
This is a DIAGRAM/FLOWCHART. Structural integrity is important.

Allowed edits:
- Add/modify LABELS on boxes or nodes
- Change COLORS of specific elements for emphasis
- Add ARROWS or CONNECTIONS between elements
- Modify TEXT within diagram components

Maintain:
- Overall FLOW and LOGIC structure
- SPATIAL RELATIONSHIPS between elements
- HIERARCHY if present

Edit instructions: {instruction}
""",
            "description": "示意图/流程图 - 保持结构完整，可修改标签和颜色"
        },
        
        ImageType.MATH_FORMULA: {
            "priority": "use_search_heavily",
            "edit_allowed": "minimal",
            "edit_scope": "specific_symbols",   
            "quality_threshold": 9.5,
            "can_use_threshold": 7.5,
            "edit_instruction_template": """
CRITICAL: This is a MATHEMATICAL FORMULA requiring PERFECT ACCURACY.

Allowed edits:
- Change specific VARIABLES (e.g., x → y, a → b)
- Modify NUMERICAL values in the formula
- Add/change SUBSCRIPTS or SUPERSCRIPTS

EXTREMELY CAREFUL:
- Mathematical expressions must remain VALID
- OPERATORS must be correct (±, ×, ÷, =, etc.)
- PARENTHESES and BRACKETS must be balanced
- LaTeX/mathematical notation must be precise

Edit instructions: {instruction}

Double-check: Any formula change must result in a mathematically valid expression.
""",
            "description": "数学公式 - 要求完美精确，仅修改特定变量或数值"
        },
        
        ImageType.MAP: {
            "priority": "use_search_preferred",
            "edit_allowed": "moderate",
            "edit_scope": "labels_and_markers",  
            "quality_threshold": 8.0,
            "can_use_threshold": 7.0,
            "edit_instruction_template": """
This is a MAP. Spatial accuracy is important.

Allowed edits:
- Add/modify LOCATION MARKERS (pins, circles)
- Change LABELS for places, roads, or regions
- Add ANNOTATIONS (distances, directions)
- Adjust COLORS for different areas/zones

Maintain:
- GEOGRAPHIC accuracy and proportions
- SPATIAL RELATIONSHIPS between locations
- SCALE and orientation

Edit instructions: {instruction}
""",
            "description": "地图 - 保持地理精确，可添加标记和标签"
        },
        
        ImageType.NATURAL_SCENE: {
            "priority": "use_search_balanced",  
            "edit_allowed": "flexible",         
            "edit_scope": "broad_changes",    
            "quality_threshold": 7.0,
            "can_use_threshold": 6.5,
            "edit_instruction_template": """
This is a NATURAL SCENE. More flexible editing is allowed.

You can:
- Add or modify OBJECTS in the scene (animals, plants, etc.)
- Change WEATHER conditions or LIGHTING
- Adjust COLORS and overall ATMOSPHERE
- Add FOREGROUND or BACKGROUND elements

Guidelines:
- Keep the scene REALISTIC and NATURAL-looking
- Maintain PERSPECTIVE and SCALE consistency
- Ensure good VISUAL COMPOSITION

Edit instructions: {instruction}
""",
            "description": "自然场景 - 可灵活修改，保持真实自然"
        },
        
        ImageType.ARTISTIC: {
            "priority": "generation_friendly",   
            "edit_allowed": "very_flexible",  
            "edit_scope": "creative",   
            "quality_threshold": 6.5,
            "can_use_threshold": 6.0,
            "edit_instruction_template": """
This is an ARTISTIC image. Creative freedom is encouraged.

You can:
- Add ARTISTIC ELEMENTS or DECORATIVE features
- Blend STYLES or TECHNIQUES
- Modify COLORS, TEXTURES, PATTERNS freely
- Transform or REMIX visual elements

Focus on:
- AESTHETIC appeal
- VISUAL INTEREST and creativity
- COHERENT overall composition

Edit instructions: {instruction}
""",
            "description": "艺术图片 - 创意自由，注重美学"
        },
        
        ImageType.EVERYDAY_OBJECT: {
            "priority": "use_search_balanced",
            "edit_allowed": "moderate",
            "edit_scope": "object_properties",  
            "quality_threshold": 7.5,
            "can_use_threshold": 6.5,
            "edit_instruction_template": """
This is an EVERYDAY OBJECT image.

Allowed edits:
- Change OBJECT COLORS or PATTERNS
- Add/remove SIMILAR OBJECTS
- Modify BACKGROUND or SETTING
- Adjust QUANTITY of objects

Maintain:
- REALISTIC appearance of objects
- PROPER PROPORTIONS and scale
- COHERENT lighting and shadows

Edit instructions: {instruction}
""",
            "description": "日常物体 - 可适度修改，保持真实"
        },
        
        ImageType.ARCHITECTURAL: {
            "priority": "use_search_preferred",
            "edit_allowed": "moderate",
            "edit_scope": "details_and_context",  
            "quality_threshold": 8.0,
            "can_use_threshold": 6.8,
            "edit_instruction_template": """
This is an ARCHITECTURAL image.

Allowed edits:
- Add/modify ARCHITECTURAL DETAILS (windows, doors, decorations)
- Change COLORS or MATERIALS (paint, cladding)
- Adjust SURROUNDING CONTEXT (landscaping, vehicles, people)
- Modify LIGHTING or TIME OF DAY

Preserve:
- Overall STRUCTURE and PROPORTIONS
- ARCHITECTURAL STYLE consistency
- PERSPECTIVE and GEOMETRY

Edit instructions: {instruction}
""",
            "description": "建筑 - 保持结构，可修改细节和背景"
        },
        
        ImageType.MIXED: {
            "priority": "use_search_balanced",
            "edit_allowed": "moderate",
            "edit_scope": "careful_mixed", 
            "quality_threshold": 7.5,
            "can_use_threshold": 6.5,
            "edit_instruction_template": """
This is a MIXED-TYPE image containing multiple elements.

Approach:
1. IDENTIFY which parts need precision (text, data, geometry)
2. IDENTIFY which parts allow flexibility (backgrounds, decorative elements)
3. Apply APPROPRIATE editing care to each part

General rules:
- Be CAUTIOUS with precise elements (numbers, shapes, medical content)
- Be more FLEXIBLE with aesthetic elements (colors, backgrounds)
- Maintain OVERALL COHERENCE

Edit instructions: {instruction}
""",
            "description": "混合类型 - 根据具体内容分区处理"
        }
    }
    
    @classmethod
    def identify_image_type(cls, plan_summary: str, question_type: str, 
                           original_question: str = "", use_gpt4o: bool = True) -> ImageType:
        
        if use_gpt4o:
            try:
                image_type_str = identify_image_type_with_gpt4o(
                    plan_summary=plan_summary,
                    question_type=question_type,
                    original_question=original_question
                )
                try:
                    return ImageType(image_type_str)
                except ValueError:
                    print(f"[identify_image_type] GPT-4o 返回无效类型: {image_type_str}, 使用关键词匹配")
            except Exception as e:
                print(f"[identify_image_type] GPT-4o 调用失败: {e}, 使用关键词匹配")
        
        combined_text = f"{plan_summary} {question_type} {original_question}".lower()
        
        type_keywords = {
            ImageType.GEOMETRIC: [
                "geometric", "geometry", "angle", "triangle", "circle", "polygon", 
                "rectangle", "square", "shape", "line", "parallel", "perpendicular",
                "几何", "角度", "三角形", "圆形", "多边形", "正方形", "矩形"
            ],
            ImageType.MEDICAL: [
                "medical", "anatomy", "x-ray", "ct scan", "mri", "ultrasound",
                "organ", "tissue", "diagnosis", "clinical", "radiology",
                "医学", "解剖", "扫描", "器官", "组织", "诊断", "临床"
            ],
            ImageType.MATH_FORMULA: [
                "formula", "equation", "mathematical expression", "algebra",
                "calculus", "integral", "derivative", "function",
                "公式", "方程", "数学表达式", "代数", "微积分", "积分", "导数", "函数"
            ],
            ImageType.CHART_GRAPH: [
                "chart", "graph", "plot", "bar chart", "line graph", "pie chart",
                "histogram", "scatter plot", "data visualization", "statistics",
                "图表", "柱状图", "折线图", "饼图", "数据可视化", "统计"
            ],
            ImageType.TEXT_DENSE: [
                "document", "text", "paragraph", "article", "page", "receipt",
                "menu", "sign", "label", "caption", "ocr",
                "文档", "文字", "段落", "文章", "页面", "菜单", "标签", "字幕"
            ],
            ImageType.DIAGRAM: [
                "diagram", "flowchart", "flow chart", "schematic", "blueprint",
                "circuit", "process", "system", "structure diagram",
                "示意图", "流程图", "电路图", "系统图", "结构图"
            ],
            ImageType.MAP: [
                "map", "geography", "location", "route", "navigation", "atlas",
                "地图", "地理", "位置", "路线", "导航"
            ],
            ImageType.ARCHITECTURAL: [
                "architecture", "building", "structure", "facade", "interior",
                "construction", "architectural", "skyscraper", "house",
                "建筑", "楼房", "结构", "外墙", "室内", "建造"
            ],
            ImageType.NATURAL_SCENE: [
                "landscape", "nature", "outdoor", "scenery", "environment",
                "forest", "mountain", "ocean", "sky", "natural",
                "风景", "自然", "户外", "环境", "森林", "山", "海洋", "天空"
            ],
            ImageType.ARTISTIC: [
                "art", "artistic", "painting", "drawing", "creative", "abstract",
                "artwork", "illustration", "design",
                "艺术", "绘画", "创意", "抽象", "插画", "设计"
            ],
            ImageType.EVERYDAY_OBJECT: [
                "object", "item", "product", "furniture", "tool", "appliance",
                "everyday", "household", "common object",
                "物体", "物品", "产品", "家具", "工具", "电器", "日常"
            ]
        }
        
        scores = {}
        for img_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[img_type] = score
        
        if scores:
            best_type = max(scores.keys(), key=lambda t: scores[t])
            return best_type
        
        if "numerical" in question_type.lower():
            if "count" in combined_text or "how many" in combined_text:
                return ImageType.EVERYDAY_OBJECT
            return ImageType.CHART_GRAPH
        
        return ImageType.MIXED
    
    @classmethod
    def get_strategy(cls, image_type: ImageType) -> Dict[str, Any]:
        return cls.STRATEGIES.get(image_type, cls.STRATEGIES[ImageType.MIXED])
    
    @classmethod
    def should_prefer_search(cls, image_type: ImageType) -> bool:
        strategy = cls.get_strategy(image_type)
        return strategy["priority"] in ["use_search_heavily", "use_search_preferred"]
    
    @classmethod
    def get_can_use_threshold(cls, image_type: ImageType) -> float:
        strategy = cls.get_strategy(image_type)
        return strategy.get("can_use_threshold", 7.0)
    
    @classmethod
    def format_edit_instruction(cls, image_type: ImageType, base_instruction: str) -> str:

        strategy = cls.get_strategy(image_type)
        template = strategy.get("edit_instruction_template", "{instruction}")
        
        return template.format(instruction=base_instruction)
    
    @classmethod
    def get_evaluation_prompt_suffix(cls, image_type: ImageType) -> str:

        strategy = cls.get_strategy(image_type)
        priority = strategy["priority"]
        threshold = strategy["can_use_threshold"]
        
        if priority == "use_search_heavily":
            return f"""
SPECIAL NOTE: This is a {image_type.value.upper()} image requiring HIGH PRECISION.
- Be MORE LENIENT in evaluation (threshold: {threshold}/10)
- PRIORITIZE using existing images over generation
- Set can_use_directly=TRUE if quality >= {threshold}/10 and image is reasonably related
- Editing this type of image carries HIGH RISK of precision loss
"""
        elif priority == "use_search_preferred":
            return f"""
NOTE: This is a {image_type.value.upper()} image where precision is important.
- Be reasonably lenient (threshold: {threshold}/10)
- PREFER using existing images when possible
- Set can_use_directly=TRUE if quality >= {threshold}/10
"""
        else:
            return ""


def demo_usage():
    
    print("="*80)
    print("图片类型智能策略系统演示")
    print("="*80)
    
    print("\n示例1: 几何图形")
    print("-"*80)
    plan1 = "Generate a question about triangle angles and parallel lines"
    img_type1 = ImageTypeStrategy.identify_image_type(plan1, "numerical", "")
    print(f"识别类型: {img_type1.value}")
    strategy1 = ImageTypeStrategy.get_strategy(img_type1)
    print(f"优先级: {strategy1['priority']}")
    print(f"编辑允许: {strategy1['edit_allowed']}")
    print(f"可用阈值: {strategy1['can_use_threshold']}")
    print(f"说明: {strategy1['description']}")
    
    base_instruction1 = "Add angle labels α, β, γ to the triangle vertices"
    formatted1 = ImageTypeStrategy.format_edit_instruction(img_type1, base_instruction1)
    print(f"\n格式化后的指令:\n{formatted1[:200]}...")
    
    print("\n\n示例2: 文字密集图片")
    print("-"*80)
    plan2 = "Generate a question about text on a restaurant menu"
    img_type2 = ImageTypeStrategy.identify_image_type(plan2, "text", "")
    print(f"识别类型: {img_type2.value}")
    strategy2 = ImageTypeStrategy.get_strategy(img_type2)
    print(f"优先级: {strategy2['priority']}")
    print(f"编辑范围: {strategy2['edit_scope']}")
    print(f"说明: {strategy2['description']}")
    
    base_instruction2 = "Change the price of 'Pasta' from $12 to $15"
    formatted2 = ImageTypeStrategy.format_edit_instruction(img_type2, base_instruction2)
    print(f"\n格式化后的指令:\n{formatted2[:300]}...")
    
    print("\n\n示例3: 自然场景")
    print("-"*80)
    plan3 = "Add some birds flying in the sky over mountains"
    img_type3 = ImageTypeStrategy.identify_image_type(plan3, "multiple choice", "")
    print(f"识别类型: {img_type3.value}")
    strategy3 = ImageTypeStrategy.get_strategy(img_type3)
    print(f"优先级: {strategy3['priority']}")
    print(f"编辑灵活度: {strategy3['edit_allowed']}")
    print(f"说明: {strategy3['description']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demo_usage()

