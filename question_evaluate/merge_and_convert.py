import json
import pandas as pd
import argparse
import os
from pathlib import Path

def merge_json_files(input_files, output_file):
    all_data = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f" 文件不存在，跳过: {file_path}")
            continue
            
        print(f"读取: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"   ✓ 加载 {len(data)} 条记录")
            all_data.extend(data)
    
    print(f"\n总计合并: {len(all_data)} 条记录")
    

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"保存到: {output_file}")
    
    return all_data

def add_default_scores(data_list, default_score=1.0, use_existing_scores=True):

    for item in data_list:
        if use_existing_scores:

            if 'score' not in item:
                item['score'] = default_score
        else:

            item['score'] = default_score
    return data_list

def convert_to_parquet(json_file, output_dir, save_name, 
                       min_score=0.0, max_score=1.0, 
                       use_existing_scores=True,
                       filter_by_score=False,
                       target_samples=512,
                       enable_补充=True):
    import random
    
    print(f"\n开始转换为parquet...")
    print(f"   - 输入: {json_file}")
    
    if filter_by_score:
        print(f"   - 根据score筛选: [{min_score}, {max_score}]")
    else:
        print(f"   - 不根据score筛选（保留所有数据）")
    
    if enable_补充:
        print(f"   - 自动补充: 启用（目标 {target_samples} 条）")
    
    if not use_existing_scores:
        print(f"   - 忽略已有score，全部设为1.0")
    else:
        print(f"   - 使用已有score（无score的默认1.0）")
    

    with open(json_file, 'r') as f:
        datas = json.load(f)
    
    print(f"   - 原始记录: {len(datas)}")
    

    datas = add_default_scores(datas, default_score=1.0, 
                              use_existing_scores=use_existing_scores)
    

    filtered_datas = []  
    excluded_datas = []     
    
    for data in datas:

        if not (data.get('answer') not in ['', 'None', None] and 'image' in data):
            continue
        
        score = data.get('score', 1.0)
        
        formatted_data = {
            'problem': data['question'],
            'answer': data['answer'],
            'score': score,
            'images': data.get('image', ''),  # base64格式
            'problem_type': data.get('question_type', 'unknown'),
            'image_type': data.get('image_type', 'unknown')  # 添加 image_type 字段
        }
        

        if filter_by_score and not (min_score <= score <= max_score):
            excluded_datas.append(formatted_data)
        else:
            filtered_datas.append(formatted_data)
    
    print(f"   - 筛选通过: {len(filtered_datas)} 条")
    

    if enable_补充 and filter_by_score and len(filtered_datas) < target_samples:
        need_count = target_samples - len(filtered_datas)
        available_count = len(excluded_datas)
        
        print(f"\n筛选数据不足 {target_samples} 条，启动自动补充...")
        print(f"   - 当前数量: {len(filtered_datas)}")
        print(f"   - 需要补充: {need_count}")
        print(f"   - 可用数据: {available_count}")
        
        if available_count > 0:

            补充数量 = min(need_count, available_count)
            random.seed(42)  # 设置随机种子以保证可重复性
            补充数据 = random.sample(excluded_datas, 补充数量)
            filtered_datas.extend(补充数据)
            
            print(f"  已补充: {补充数量} 条")
            print(f"   - 最终数量: {len(filtered_datas)}")
            

            补充scores = [d['score'] for d in 补充数据]
            print(f"   - 补充数据score范围: [{min(补充scores):.3f}, {max(补充scores):.3f}]")
        else:
            print(f"无可用数据补充（总数据不足）")
    
    print(f"\n   - 最终有效记录: {len(filtered_datas)} 条")
    
    if not filtered_datas:
        print("没有有效数据（缺少答案或图片），停止转换")
        return None
    

    os.makedirs(output_dir, exist_ok=True)
    

    output_path = os.path.join(output_dir, f"{save_name}_train.parquet")
    df = pd.DataFrame(filtered_datas)
    df.to_parquet(output_path, index=False)
    print(f"Parquet文件保存到: {output_path}")
    

    summary_path = output_path.replace('.parquet', '_summary.json')
    summary = {
        "total_samples": len(filtered_datas),
        "score_range": [min_score, max_score],
        "experiment_name": save_name,
        "output_file": output_path,
        "source_files": [str(json_file)]
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"摘要保存到: {summary_path}")
    

    print(f"\n数据统计:")
    print(f"   - 总样本数: {len(filtered_datas)}")
    print(f"   - 问题类型分布:")
    type_counts = df['problem_type'].value_counts()
    for ptype, count in type_counts.items():
        print(f"     • {ptype}: {count}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='合并JSON文件并转换为parquet')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='输入的JSON文件路径（可以多个）')
    parser.add_argument('--output_dir', type=str, 
                        default='path_to_local_parquet',
                        help='Parquet输出目录')
    parser.add_argument('--save_name', type=str, default='merged_generated',
                        help='输出文件名前缀')
    parser.add_argument('--min_score', type=float, default=0.0,
                        help='最低评分阈值（仅当--filter_by_score时生效）')
    parser.add_argument('--max_score', type=float, default=1.0,
                        help='最高评分阈值（仅当--filter_by_score时生效）')
    parser.add_argument('--filter_by_score', action='store_true',
                        help='根据score筛选数据')
    parser.add_argument('--ignore_existing_scores', action='store_true',
                        help='忽略已有score，全部设为1.0')
    parser.add_argument('--merged_json', type=str, default=None,
                        help='合并后的JSON文件保存路径（可选）')
    parser.add_argument('--target_samples', type=int, default=512,
                        help='目标样本数量（默认512，用于自动补充）')
    parser.add_argument('--enable_补充', action='store_true', default=True,
                        help='启用自动补充到目标数量（默认启用）')
    parser.add_argument('--disable_补充', dest='enable_补充', action='store_false',
                        help='禁用自动补充')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("合并和转换工具")
    print("=" * 80)
    

    if args.merged_json is None:

        args.merged_json = os.path.join(
            args.output_dir, 
            f"{args.save_name}_merged.json"
        )
    
    merged_data = merge_json_files(args.input_files, args.merged_json)
    
    if not merged_data:
        print("没有数据可以转换")
        return
    

    parquet_path = convert_to_parquet(
        json_file=args.merged_json,
        output_dir=args.output_dir,
        save_name=args.save_name,
        min_score=args.min_score,
        max_score=args.max_score,
        use_existing_scores=not args.ignore_existing_scores,
        filter_by_score=args.filter_by_score,
        target_samples=args.target_samples,
        enable_补充=args.enable_补充
    )
    
    if parquet_path:
        print("\n" + "=" * 80)
        print("转换完成！")
        print("=" * 80)
        print(f"Parquet文件: {parquet_path}")
        print(f"合并JSON: {args.merged_json}")
    else:
        print("\n转换失败")

if __name__ == '__main__':
    main()

