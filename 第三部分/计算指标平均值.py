#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算十次症状总结指标值的平均值

功能：
1. 读取Excel文件中的指标数据
2. 计算每个案例每个指标的9次比较平均值
3. 计算每个案例四个指标的平均值
4. 输出结果到新的Excel文件
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

# 配置日志记录
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_excel_data(file_path: str) -> pd.DataFrame:
    """
    读取Excel文件数据
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        pandas DataFrame包含所有数据
    """
    logger.info(f"开始读取Excel文件: {file_path}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        logger.info(f"成功读取Excel文件，数据形状: {df.shape}")
        logger.info(f"列名: {list(df.columns)}")
        
        # 显示前几行数据以了解结构
        logger.info("数据预览:")
        logger.info(df.head())
        
        return df
    except Exception as e:
        logger.error(f"读取Excel文件失败: {e}")
        raise

def analyze_data_structure(df: pd.DataFrame) -> Dict:
    """
    分析数据结构，识别案例和指标
    
    Args:
        df: 包含数据的DataFrame
        
    Returns:
        包含数据结构信息的字典
    """
    logger.info("开始分析数据结构")
    
    structure_info = {
        'columns': list(df.columns),
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict()
    }
    
    # 识别可能的案例列和指标列
    case_columns = []
    metric_columns = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        # 识别案例相关的列
        if any(keyword in col_lower for keyword in ['案例', 'case', '样本', 'sample', '编号', 'id']):
            case_columns.append(col)
        # 识别指标相关的列
        elif any(keyword in col_lower for keyword in ['指标', 'metric', '评分', 'score', '值', 'value']):
            metric_columns.append(col)
    
    structure_info['case_columns'] = case_columns
    structure_info['metric_columns'] = metric_columns
    
    logger.info(f"识别到的案例列: {case_columns}")
    logger.info(f"识别到的指标列: {metric_columns}")
    
    return structure_info

def calculate_averages(df: pd.DataFrame, structure_info: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算平均值
    
    Args:
        df: 原始数据
        structure_info: 数据结构信息
        
    Returns:
        Tuple[每个指标的平均值DataFrame, 每个案例的总平均值DataFrame]
    """
    logger.info("开始计算平均值")
    
    # 如果没有明确识别到案例列，使用第一列作为案例标识
    if not structure_info['case_columns']:
        case_column = df.columns[0]
        logger.warning(f"未识别到案例列，使用第一列作为案例标识: {case_column}")
    else:
        case_column = structure_info['case_columns'][0]
    
    # 识别指标列 - 如果没有明确识别，假设所有数值列都是指标
        if not structure_info['metric_columns']:
            metric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除案例列（如果是数值型）
            if case_column in metric_columns:
                metric_columns.remove(case_column)
            # 排除比较次数列，因为它不是评估指标
            if '比较次数' in metric_columns:
                metric_columns.remove('比较次数')
            logger.warning(f"未识别到指标列，使用数值列作为指标（排除比较次数）: {metric_columns}")
        else:
            metric_columns = structure_info['metric_columns']
    
    # 确保指标列都是数值型
    metric_columns = [col for col in metric_columns if col in df.select_dtypes(include=[np.number]).columns]
    
    if not metric_columns:
        raise ValueError("未找到有效的数值型指标列")
    
    logger.info(f"使用的案例列: {case_column}")
    logger.info(f"使用的指标列: {metric_columns}")
    
    # 计算每个案例每个指标的9次比较平均值
    # 假设数据已经按照9次比较组织，每个案例有9行数据
    case_metric_avg = df.groupby(case_column)[metric_columns].mean()
    case_metric_avg.columns = [f"{col}_平均值" for col in metric_columns]
    
    # 计算每个案例四个指标的平均值
    case_total_avg = case_metric_avg.mean(axis=1)
    case_total_avg.name = "案例总平均值"
    
    # 合并结果
    result_df = pd.concat([case_metric_avg, case_total_avg], axis=1)
    
    logger.info("平均值计算完成")
    logger.info(f"结果数据形状: {result_df.shape}")
    
    return case_metric_avg, case_total_avg

def save_results(metric_avg_df: pd.DataFrame, total_avg_series: pd.Series, output_path: str):
    """
    保存计算结果到Excel文件
    
    Args:
        metric_avg_df: 每个指标的平均值DataFrame
        total_avg_series: 每个案例的总平均值Series
        output_path: 输出文件路径
    """
    logger.info(f"开始保存结果到: {output_path}")
    
    # 创建Excel写入器
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 保存每个指标的平均值
        metric_avg_df.to_excel(writer, sheet_name='指标平均值', index=True)
        
        # 保存每个案例的总平均值
        total_avg_df = pd.DataFrame({
            '案例': total_avg_series.index,
            '总平均值': total_avg_series.values
        })
        total_avg_df.to_excel(writer, sheet_name='案例总平均值', index=False)
        
        # 保存详细统计信息
        stats_df = pd.DataFrame({
            '统计项': ['案例数量', '指标数量', '平均案例总平均值'],
            '值': [len(total_avg_series), len(metric_avg_df.columns), total_avg_series.mean()]
        })
        stats_df.to_excel(writer, sheet_name='统计信息', index=False)
    
    logger.info("结果保存完成")

def main():
    """主函数"""
    # 文件路径
    input_file = r"d:\大三下学习\科研路\自己的方向\文章结果\第三部分\十次症状总结指标值.xlsx"
    output_file = r"d:\大三下学习\科研路\自己的方向\文章结果\第三部分\指标平均值计算结果.xlsx"
    
    try:
        # 1. 读取数据
        df = read_excel_data(input_file)
        
        # 2. 分析数据结构
        structure_info = analyze_data_structure(df)
        
        # 3. 计算平均值
        metric_avg_df, total_avg_series = calculate_averages(df, structure_info)
        
        # 4. 保存结果
        save_results(metric_avg_df, total_avg_series, output_file)
        
        # 5. 打印结果摘要
        logger.info("\n=== 计算结果摘要 ===")
        logger.info(f"案例数量: {len(total_avg_series)}")
        logger.info(f"指标数量: {len(metric_avg_df.columns)}")
        logger.info(f"案例总平均值范围: {total_avg_series.min():.4f} - {total_avg_series.max():.4f}")
        logger.info(f"平均案例总平均值: {total_avg_series.mean():.4f}")
        
        logger.info("\n每个指标的平均值:")
        for col in metric_avg_df.columns:
            logger.info(f"{col}: {metric_avg_df[col].mean():.4f}")
        
        logger.info(f"\n详细结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()