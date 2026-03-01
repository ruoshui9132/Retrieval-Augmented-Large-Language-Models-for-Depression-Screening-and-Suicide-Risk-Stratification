#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本文件转Excel表格转换器
将标准病例案例信息.txt转换为Excel格式
"""

import re
import pandas as pd
from pathlib import Path

def parse_text_file(file_path):
    """
    解析文本文件，提取病例信息
    
    Args:
        file_path (str): 文本文件路径
        
    Returns:
        list: 包含所有病例信息的字典列表
    """
    cases = []
    current_case = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return []
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检查是否是新的病例开始（id=数字）
        if re.match(r'^id=\d+$', line):
            # 如果当前病例不为空，先保存
            if current_case:
                cases.append(current_case)
            # 开始新的病例
            current_case = {'id': line.split('=')[1]}
        
        # 解析基本信息
        elif line.startswith('基本信息：'):
            current_case['基本信息'] = line[5:]  # 去掉"基本信息："前缀
        
        # 解析背景
        elif line.startswith('背景：'):
            current_case['背景'] = line[3:]  # 去掉"背景："前缀
        
        # 解析临床症状
        elif line.startswith('临床症状：'):
            current_case['临床症状'] = line[5:]  # 去掉"临床症状："前缀
        
        # 解析检查
        elif line.startswith('检查：') or line.startswith('检查结果：'):
            # 处理两种可能的检查前缀
            if line.startswith('检查：'):
                current_case['检查'] = line[3:]  # 去掉"检查："前缀
            else:
                current_case['检查'] = line[5:]  # 去掉"检查结果："前缀
        
        # 如果行不是以任何前缀开头，可能是多行内容的延续
        elif current_case and '检查' in current_case:
            # 如果是检查字段的多行内容
            current_case['检查'] += ' ' + line
        elif current_case and '临床症状' in current_case:
            # 如果是临床症状字段的多行内容
            current_case['临床症状'] += ' ' + line
        elif current_case and '背景' in current_case:
            # 如果是背景字段的多行内容
            current_case['背景'] += ' ' + line
        elif current_case and '基本信息' in current_case:
            # 如果是基本信息字段的多行内容
            current_case['基本信息'] += ' ' + line
    
    # 添加最后一个病例
    if current_case:
        cases.append(current_case)
    
    return cases

def save_to_excel(cases, output_path):
    """
    将病例数据保存为Excel文件
    
    Args:
        cases (list): 病例数据列表
        output_path (str): 输出Excel文件路径
    """
    if not cases:
        print("没有数据可保存")
        return False
    
    try:
        # 创建DataFrame
        df = pd.DataFrame(cases)
        
        # 确保列的顺序：id, 基本信息, 背景, 临床症状, 检查
        column_order = ['id', '基本信息', '背景', '临床症状', '检查']
        # 只保留存在的列
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # 保存为Excel
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"成功转换 {len(cases)} 个病例到 {output_path}")
        return True
        
    except Exception as e:
        print(f"保存Excel文件时出错：{e}")
        return False

def main():
    """主函数"""
    # 文件路径
    input_file = r"d:\大三下学习\科研路\自己的方向\文章结果\第二部分\调整原数据集\标准病例案例信息.txt"
    output_file = r"d:\大三下学习\科研路\自己的方向\文章结果\第二部分\调整原数据集\标准病例案例信息.xlsx"
    
    print("开始解析文本文件...")
    
    # 解析文本文件
    cases = parse_text_file(input_file)
    
    if not cases:
        print("没有找到有效的病例数据")
        return
    
    print(f"成功解析 {len(cases)} 个病例")
    
    # 显示前几个病例作为示例
    print("\n前3个病例的示例：")
    for i, case in enumerate(cases[:3]):
        print(f"\n病例 {i+1}:")
        for key, value in case.items():
            # 截断长文本以便显示
            display_value = value[:100] + "..." if len(value) > 100 else value
            print(f"  {key}: {display_value}")
    
    # 保存为Excel
    print(f"\n正在保存到Excel文件: {output_file}")
    success = save_to_excel(cases, output_file)
    
    if success:
        print("转换完成！")
    else:
        print("转换失败")

if __name__ == "__main__":
    main()