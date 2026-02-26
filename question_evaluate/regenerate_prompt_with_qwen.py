import json
import os
import sys
from pathlib import Path
from collections import defaultdict


sys.path.append(str(Path(__file__).parent.parent))

from question_generate.api_clients import Qwen3VLClient


IMAGE_TYPES = [
    "geometric", "medical", "chart_graph", "text_dense",
    "diagram", "math_formula", "map", "natural_scene",
    "artistic", "everyday_object", "architectural", "mixed"
]


def aggregate_by_imagetype(results):
    stats = {}
    
    for img_type in IMAGE_TYPES:
        stats[img_type] = {
            'total_evaluated': 0,
            'correct': 0,
            'accuracy': 0.0,
            'errors': []
        }
    

    for result in results:
        img_type = result.get('image_type', 'mixed')
        
        if img_type not in stats:
            img_type = 'mixed'
        
        stats[img_type]['total_evaluated'] += 1
        
        if result['is_correct']:
            stats[img_type]['correct'] += 1
        else:
            stats[img_type]['errors'].append({
                'error_type': result.get('error_type', 'other'),
                'error_description': result.get('error_description', '')
            })
    

    for img_type in stats:
        total = stats[img_type]['total_evaluated']
        if total > 0:
            stats[img_type]['accuracy'] = stats[img_type]['correct'] / total
    

    for img_type in stats:
        error_by_type = defaultdict(list)
        for error in stats[img_type]['errors']:
            error_type = error['error_type']
            error_desc = error['error_description'][:200] 
            error_by_type[error_type].append(error_desc)
        stats[img_type]['error_descriptions_by_type'] = dict(error_by_type)
        del stats[img_type]['errors']  
    

    stats = {k: v for k, v in stats.items() if v['total_evaluated'] > 0}
    
    return stats


def analyze_weakness_with_qwen(
    img_type: str, 
    error_descriptions: list, 
    error_type: str, 
    qwen_client: Qwen3VLClient
) -> dict:

    import re
    

    sample_errors = error_descriptions[:5]  # 只取前5个作为示例
    errors_text = "\n".join([f"- {err}" for err in sample_errors])
    
    prompt = f"""You are analyzing errors from a vision-language model.

Image Type: {img_type}
Error Type: {error_type}
Error Examples:
{errors_text}

Task: Based on these errors, provide guidance for generating new training questions.

Output ONLY this JSON (no extra text):
{{
    "generation_direction": "Specific, actionable guidance on what kinds of questions to generate to fix this weakness (2-3 sentences)"
}}"""
    
    max_retries = 3
    for retry in range(max_retries):
        try:

            messages = [{"role": "user", "content": prompt}]
            response = qwen_client.chat(
                messages=messages,
                max_tokens=400
            )
            
            if response:

                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group())
                        if 'generation_direction' in analysis:
                            return analysis
                    except json.JSONDecodeError:
                        pass
            
            if retry < max_retries - 1:
                import time
                time.sleep(2)
                print(f"    [Qwen3-vl Analysis] Retry {retry + 1}/{max_retries}")
                continue
        except Exception as e:
            if retry < max_retries - 1:
                import time
                time.sleep(2)
                print(f"    [Qwen3-vl Analysis] Error: {e}, retry {retry + 1}/{max_retries}")
                continue
    

    print(f"    [Qwen3-vl Analysis] Failed, using fallback")
    return {
        'generation_direction': f'Generate {img_type} questions focusing on {error_type} scenarios with varying difficulty levels'
    }


def generate_weakness_prompt(stats, qwen_client: Qwen3VLClient):
    lines = ["[Model Weakness Analysis - Targeted Generation Guidance]", ""]
    

    weak_types = []  
    moderate_types = []  
    strong_types = []   
    
    for img_type, data in stats.items():
        accuracy = data['accuracy']
        if accuracy < 0.70:
            weak_types.append((img_type, data))
        elif accuracy < 0.85:
            moderate_types.append((img_type, data))
        else:
            strong_types.append((img_type, data))
    

    weak_types.sort(key=lambda x: x[1]['accuracy'])
    moderate_types.sort(key=lambda x: x[1]['accuracy'])
    strong_types.sort(key=lambda x: x[1]['accuracy'])
    

    if weak_types:
        lines.append("=== HIGH PRIORITY: Types Requiring Significant Improvement ===")
        lines.append("")
        for img_type, data in weak_types:
            lines.append(f"[{img_type.upper()}] (Accuracy: {data['accuracy']*100:.1f}%, {data['total_evaluated']} samples)")
            

            if data['error_descriptions_by_type']:
                lines.append(f"  Generation Directions:")

                sorted_errors = sorted(
                    data['error_descriptions_by_type'].items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )
                
                for error_type, descriptions in sorted_errors:
                    count = len(descriptions)
                    total_errors = sum(len(v) for v in data['error_descriptions_by_type'].values())
                    pct = count / total_errors * 100 if total_errors > 0 else 0
                    

                    print(f"  Analyzing {img_type} - {error_type} ({count} cases) with Qwen3-vl...")
                    analysis = analyze_weakness_with_qwen(img_type, descriptions, error_type, qwen_client)
                    
                    lines.append(f"    [{error_type}] ({count} cases, {pct:.0f}%):")
                    lines.append(f"      → {analysis['generation_direction']}")
            
            lines.append("")
    

    if moderate_types:
        lines.append("=== MODERATE PRIORITY: Types Needing Consolidation ===")
        lines.append("")
        for img_type, data in moderate_types:
            lines.append(f"[{img_type.upper()}] (Accuracy: {data['accuracy']*100:.1f}%, {data['total_evaluated']} samples)")
            

            if data['error_descriptions_by_type']:
                main_error_type = max(data['error_descriptions_by_type'].items(), key=lambda x: len(x[1]))
                error_type, descriptions = main_error_type
                count = len(descriptions)
                

                print(f"  Analyzing {img_type} - {error_type} ({count} cases) with Qwen3-vl...")
                analysis = analyze_weakness_with_qwen(img_type, descriptions, error_type, qwen_client)
                
                lines.append(f"  → {analysis['generation_direction']}")
            
            lines.append("")
    

    if strong_types:
        lines.append("=== LOW PRIORITY: Well-Mastered Types ===")
        lines.append("")
        for img_type, data in strong_types:
            lines.append(f"[{img_type.upper()}] (Accuracy: {data['accuracy']*100:.1f}%, {data['total_evaluated']} samples)")
            lines.append(f"  → Reduce generation for this type, allocate resources to weaker areas")
            lines.append("")
    
    return "\n".join(lines)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python regenerate_prompt_with_qwen.py <detailed_results.jsonl>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    

    print(f"Loading results from {results_file}...")
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    print(f"Loaded {len(results)} results")
    

    print("Aggregating by image type...")
    stats = aggregate_by_imagetype(results)
    

    print("\nInitializing Qwen3-vl client...")
    qwen_client = Qwen3VLClient(model_name="qwen-vl-max")
    

    print("\nGenerating weakness prompt with Qwen3-vl analysis...")
    weakness_prompt = generate_weakness_prompt(stats, qwen_client)
    

    output_dir = os.path.dirname(results_file)
    prompt_file = os.path.join(output_dir, "weakness_prompt.txt")
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(weakness_prompt)
    
    print(f"\nWeakness prompt saved to: {prompt_file}")
    print("\n" + "="*60)
    print(weakness_prompt)
    print("="*60)


if __name__ == "__main__":
    main()

