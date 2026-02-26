import os
import json
import argparse
from typing import Optional
from weakness_analysis import WeaknessAnalysisTable


def list_weaknesses(table: WeaknessAnalysisTable, question_type: Optional[str] = None):
    print("\n" + "="*80)
    print("弱点列表")
    print("="*80)
    
    if question_type:

        weaknesses = table.get_weaknesses_by_type(question_type)
        if not weaknesses:
            print(f"类型 '{question_type}' 没有记录的弱点")
            return
        
        print(f"\n【{question_type}】- 共 {len(weaknesses)} 个弱点\n")
        for i, w in enumerate(weaknesses):
            print(f"ID: {i}")
            print(f"  描述: {w.get('weakness_description', 'N/A')}")
            print(f"  错误类型: {w.get('error_type', 'N/A')}")
            print(f"  出现次数: {w.get('count', 0)}")
            print(f"  示例数: {len(w.get('examples', []))}")
            print()
    else:

        if not table.weaknesses:
            print("弱点表为空")
            return
        
        for q_type, weaknesses in table.weaknesses.items():
            print(f"\n【{q_type}】- 共 {len(weaknesses)} 个弱点\n")
            for i, w in enumerate(weaknesses):
                print(f"  ID: {i}")
                print(f"    描述: {w.get('weakness_description', 'N/A')}")
                print(f"    错误类型: {w.get('error_type', 'N/A')}")
                print(f"    出现次数: {w.get('count', 0)}")
                print()


def add_weakness_interactive(table: WeaknessAnalysisTable):
    print("\n" + "="*80)
    print("添加弱点")
    print("="*80)
    
    question_type = input("问题类型: ").strip()
    weakness_desc = input("弱点描述: ").strip()
    error_type = input("错误类型: ").strip()
    question_id = input("问题ID（可选）: ").strip() or "manual_entry"
    
    weakness_info = {
        'question_id': question_id,
        'weakness_description': weakness_desc,
        'error_type': error_type,
        'timestamp': None
    }
    
    table.add_weakness(question_type, weakness_info)
    table.save()
    print("弱点已添加")


def remove_weakness_interactive(table: WeaknessAnalysisTable):
    print("\n" + "="*80)
    print("删除弱点")
    print("="*80)
    

    list_weaknesses(table)
    
    question_type = input("\n请输入问题类型: ").strip()
    if question_type not in table.weaknesses:
        print(f"类型 '{question_type}' 不存在")
        return
    
    try:
        weakness_id = int(input("请输入要删除的弱点ID: ").strip())
        table.remove_weakness(question_type, weakness_id)
        table.save()
    except ValueError:
        print("无效的ID")


def update_weakness_interactive(table: WeaknessAnalysisTable):
    print("\n" + "="*80)
    print(" 更新弱点")
    print("="*80)
    

    list_weaknesses(table)
    
    question_type = input("\n请输入问题类型: ").strip()
    if question_type not in table.weaknesses:
        print(f"类型 '{question_type}' 不存在")
        return
    
    try:
        weakness_id = int(input("请输入要更新的弱点ID: ").strip())
        
        print("\n请输入新的信息（留空表示不修改）:")
        new_desc = input("新的弱点描述: ").strip()
        new_error_type = input("新的错误类型: ").strip()
        
        new_info = {}
        if new_desc:
            new_info['weakness_description'] = new_desc
        if new_error_type:
            new_info['error_type'] = new_error_type
        
        if new_info:
            table.update_weakness(question_type, weakness_id, new_info)
            table.save()
        else:
            print("没有提供更新信息")
    
    except ValueError:
        print("无效的ID")


def export_to_markdown(table: WeaknessAnalysisTable, output_path: str):
    print(f"\n导出为Markdown: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 模型弱点分析报告\n\n")
        
        summary = table.get_summary()
        f.write(f"## 总览\n\n")
        f.write(f"- 问题类型数: {summary['total_types']}\n\n")
        
        for q_type, info in summary['by_type'].items():
            f.write(f"## {q_type}\n\n")
            f.write(f"- 弱点种类: {info['weakness_count']}\n")
            f.write(f"- 错误总数: {info['total_errors']}\n\n")
            
            f.write("### 主要弱点\n\n")
            f.write("| 排名 | 弱点描述 | 错误类型 | 出现次数 |\n")
            f.write("|------|----------|----------|----------|\n")
            
            for i, w in enumerate(info['top_weaknesses'], 1):
                desc = w.get('weakness_description', 'N/A')
                error_type = w.get('error_type', 'N/A')
                count = w.get('count', 0)
                f.write(f"| {i} | {desc} | {error_type} | {count} |\n")
            
            f.write("\n")
            

            weaknesses = table.get_weaknesses_by_type(q_type)
            if len(weaknesses) > 3:
                f.write("### 所有弱点\n\n")
                for i, w in enumerate(weaknesses):
                    desc = w.get('weakness_description', 'N/A')
                    error_type = w.get('error_type', 'N/A')
                    count = w.get('count', 0)
                    f.write(f"{i+1}. **{desc}** ({error_type}, 出现{count}次)\n")
                f.write("\n")
    
    print("导出完成")


def interactive_mode(table_path: str):
    table = WeaknessAnalysisTable(table_path)
    
    while True:
        print("\n" + "="*80)
        print("🔧 弱点分析表管理")
        print("="*80)
        print("1. 查看弱点摘要")
        print("2. 列出所有弱点")
        print("3. 列出特定类型弱点")
        print("4. 添加弱点")
        print("5. 删除弱点")
        print("6. 更新弱点")
        print("7. 导出为Markdown")
        print("0. 退出")
        print("="*80)
        
        choice = input("\n请选择操作 (0-7): ").strip()
        
        if choice == '0':
            print("再见！")
            break
        elif choice == '1':
            table.print_summary()
        elif choice == '2':
            list_weaknesses(table)
        elif choice == '3':
            q_type = input("请输入问题类型: ").strip()
            list_weaknesses(table, q_type)
        elif choice == '4':
            add_weakness_interactive(table)
        elif choice == '5':
            remove_weakness_interactive(table)
        elif choice == '6':
            update_weakness_interactive(table)
        elif choice == '7':
            output_path = input("请输入输出文件路径 (default: weakness_report.md): ").strip()
            if not output_path:
                output_path = "weakness_report.md"
            export_to_markdown(table, output_path)
        else:
            print("无效的选择")


def main():
    parser = argparse.ArgumentParser(description="弱点分析表管理工具")
    parser.add_argument('--table_path', type=str, required=True,
                        help='弱点分析表路径')
    parser.add_argument('--action', type=str, choices=['list', 'add', 'remove', 'update', 'summary', 'export', 'interactive'],
                        default='interactive',
                        help='操作类型')
    parser.add_argument('--question_type', type=str,
                        help='问题类型（用于list操作）')
    parser.add_argument('--output', type=str,
                        help='输出文件路径（用于export操作）')
    
    args = parser.parse_args()
    
    if args.action == 'interactive':
        interactive_mode(args.table_path)
    else:
        table = WeaknessAnalysisTable(args.table_path)
        
        if args.action == 'list':
            list_weaknesses(table, args.question_type)
        elif args.action == 'summary':
            table.print_summary()
        elif args.action == 'export':
            output_path = args.output or 'weakness_report.md'
            export_to_markdown(table, output_path)
        elif args.action == 'add':
            add_weakness_interactive(table)
        elif args.action == 'remove':
            remove_weakness_interactive(table)
        elif args.action == 'update':
            update_weakness_interactive(table)


if __name__ == '__main__':
    main()

