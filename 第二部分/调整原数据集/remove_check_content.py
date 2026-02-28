#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除标准病例案例信息.txt文件中每个案例的检查内容

功能说明：
- 读取原始文本文件
- 删除每个病例中的检查内容（包括"检查："和"检查结果："字段）
- 保留其他所有信息（id、基本信息、背景、临床症状）
- 生成新的文本文件

作者：AI助手
日期：2025-01-20
"""

import re
import os

def remove_check_content(input_file_path, output_file_path):
    """
    删除病例文件中的检查内容
    
    参数：
    input_file_path: 输入文件路径
    output_file_path: 输出文件路径
    
    返回：
    bool: 处理是否成功
    """
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"错误：输入文件不存在 - {input_file_path}")
            return False
        
        # 读取原始文件内容
        with open(input_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        print(f"成功读取文件，文件大小：{len(content)} 字符")
        
        # 使用正则表达式匹配病例模式
        # 每个病例以"id=数字"开头，到下一个"id=数字"或文件结束
        case_pattern = r'(id=\d+.*?)(?=id=\d+|\Z)'
        cases = re.findall(case_pattern, content, re.DOTALL)
        
        print(f"找到 {len(cases)} 个病例")
        
        processed_cases = []
        
        for i, case in enumerate(cases, 1):
            # 删除检查内容
            # 匹配"检查："或"检查结果："及其后面的所有内容，直到下一个字段或病例结束
            case_without_check = re.sub(r'\n检查(结果)?：.*?(?=\nid=|\Z)', '', case, flags=re.DOTALL)
            
            # 确保病例以换行符结束，保持格式整洁
            case_without_check = case_without_check.rstrip() + '\n\n'
            
            processed_cases.append(case_without_check)
            
            if i <= 5:  # 显示前5个病例的处理结果作为示例
                print(f"病例 {i} 处理完成")
        
        # 将所有处理后的病例合并
        new_content = ''.join(processed_cases)
        
        # 写入新文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print(f"处理完成！新文件已保存至：{output_file_path}")
        print(f"原始文件大小：{len(content)} 字符")
        print(f"新文件大小：{len(new_content)} 字符")
        print(f"删除了约 {len(content) - len(new_content)} 字符的检查内容")
        
        return True
        
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")
        return False

def main():
    """主函数"""
    
    # 文件路径
    input_file = r"d:\大三下学习\科研路\自己的方向\文章结果\第二部分\调整原数据集\标准病例案例信息.txt"
    output_file = r"d:\大三下学习\科研路\自己的方向\文章结果\第二部分\调整原数据集\标准病例案例信息_无检查.txt"
    
    print("开始删除病例文件中的检查内容...")
    print("=" * 50)
    
    # 执行处理
    success = remove_check_content(input_file, output_file)
    
    print("=" * 50)
    if success:
        print("处理成功完成！")
    else:
        print("处理失败，请检查错误信息。")

if __name__ == "__main__":
    main()