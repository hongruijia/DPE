import argparse
from weakness_analysis import WeaknessAnalysisTable


def main():
    parser = argparse.ArgumentParser(
        description="从弱点分析表生成用于问题生成的提示字符串"
    )
    parser.add_argument(
        '--weakness_table',
        type=str,
        required=True,
        help='弱点分析表路径（weakness_analysis.json）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（默认：同目录下的weakness_prompt.txt）'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=5,
        help='每个类型显示的最大弱点数（默认：5）'
    )
    parser.add_argument(
        '--min_count',
        type=int,
        default=2,
        help='最小出现次数阈值（默认：2，过滤出现次数太少的弱点）'
    )
    parser.add_argument(
        '--print_only',
        action='store_true',
        help='只打印到屏幕，不保存文件'
    )
    
    args = parser.parse_args()
    

    print(f"加载弱点分析表: {args.weakness_table}")
    table = WeaknessAnalysisTable(args.weakness_table)
    
    if not table.weaknesses:
        print("弱点分析表为空")
        return
    

    prompt_str = table.generate_weakness_prompt_string(
        top_n=args.top_n,
        min_count=args.min_count
    )
    

    print("\n" + "="*80)
    print("生成的弱点提示字符串")
    print("="*80)
    print(prompt_str)
    print("="*80)
    

    if not args.print_only:
        if args.output is None:

            import os
            output_path = os.path.join(
                os.path.dirname(args.weakness_table),
                "weakness_prompt.txt"
            )
        else:
            output_path = args.output
        
        table.export_weakness_prompt_file(
            output_path,
            top_n=args.top_n,
            min_count=args.min_count
        )
        
        print(f"\n已保存到: {output_path}")
        print(f"\n使用建议:")
        print(f"   在问题生成时，可以将此文件内容添加到prompt中，例如:")
        print(f"   \"\"\"")
        print(f"   根据以下模型弱点，生成针对性的问题来改进模型：")
        print(f"   ")
        print(f"   {prompt_str[:200]}...")
        print(f"   \"\"\"")


if __name__ == '__main__':
    main()


















