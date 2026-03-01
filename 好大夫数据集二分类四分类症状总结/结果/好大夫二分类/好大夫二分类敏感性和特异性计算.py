import pandas as pd
import os

# 获取当前目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取目录下所有Excel文件
excel_files = [f for f in os.listdir(current_dir) if f.endswith('.xlsx')]

# 创建一个列表来存储所有文件的结果
results = []

# 遍历所有Excel文件
for file_name in excel_files:
    try:
        # 构建完整的文件路径
        file_path = os.path.join(current_dir, file_name)
        
        # 打印当前处理的文件名
        print(f"\n======== 处理文件: {file_name} ========")
        
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 计算抑郁症敏感性
        # 抑郁症敏感性 = (抑郁症为'有'且有无抑郁症为'有'的案例数) / (抑郁症为'有'的案例数)
        depression_true_positive = len(df[(df['抑郁症'] == '有') & (df['有无抑郁症'] == '有')])
        depression_positive = len(df[df['抑郁症'] == '有'])
        depression_sensitivity = depression_true_positive / depression_positive if depression_positive > 0 else 0
        
        # 计算抑郁症特异性
        # 抑郁症特异性 = (抑郁症为'无'且有无抑郁症为'无'的案例数) / (抑郁症为'无'的案例数)
        depression_true_negative = len(df[(df['抑郁症'] == '无') & (df['有无抑郁症'] == '无')])
        depression_negative = len(df[df['抑郁症'] == '无'])
        depression_specificity = depression_true_negative / depression_negative if depression_negative > 0 else 0
        
        # 计算自杀风险敏感性
        # 自杀风险敏感性 = (自杀风险为1或2或3且有无自杀风险为'有'的案例数) / (自杀风险为1或2或3的案例数)
        suicide_true_positive = len(df[(df['自杀风险'].isin([1, 2, 3])) & (df['有无自杀风险'] == '有')])
        suicide_positive = len(df[df['自杀风险'].isin([1, 2, 3])])
        suicide_sensitivity = suicide_true_positive / suicide_positive if suicide_positive > 0 else 0
        
        # 计算自杀风险特异性
        # 自杀风险特异性 = (自杀风险为0且有无自杀风险为'无'的案例数) / (自杀风险为0的案例数)
        suicide_true_negative = len(df[(df['自杀风险'] == 0) & (df['有无自杀风险'] == '无')])
        suicide_negative = len(df[df['自杀风险'] == 0])
        suicide_specificity = suicide_true_negative / suicide_negative if suicide_negative > 0 else 0
        
        # 输出统计结果
        print("\n抑郁症统计:")
        print(f"抑郁症为'有'的案例总数: {depression_positive}")
        print(f"抑郁症为'有'且有无抑郁症为'有'的案例数: {depression_true_positive}")
        print(f"抑郁症敏感性: {depression_sensitivity*100:.2f}%")
        
        print(f"\n抑郁症为'无'的案例总数: {depression_negative}")
        print(f"抑郁症为'无'且有无抑郁症为'无'的案例数: {depression_true_negative}")
        print(f"抑郁症特异性: {depression_specificity*100:.2f}%")
        
        print("\n自杀风险统计:")
        print(f"自杀风险为'1'、'2'或'3'的案例总数: {suicide_positive}")
        print(f"自杀风险为'1'、'2'或'3'且有无自杀风险为'有'的案例数: {suicide_true_positive}")
        print(f"自杀风险敏感性: {suicide_sensitivity*100:.2f}%")
        
        print(f"\n自杀风险为'0'的案例总数: {suicide_negative}")
        print(f"自杀风险为'0'且有无自杀风险为'无'的案例数: {suicide_true_negative}")
        print(f"自杀风险特异性: {suicide_specificity*100:.2f}%")
        
        print("======================================\n")
        
        # 将当前文件的结果添加到结果列表中
        results.append({
            '文件名': file_name,
            '抑郁症敏感性(%)': depression_sensitivity * 100,
            '抑郁症特异性(%)': depression_specificity * 100,
            '自杀风险敏感性(%)': suicide_sensitivity * 100,
            '自杀风险特异性(%)': suicide_specificity * 100
        })
        
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {str(e)}")
        print("======================================\n")

# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results)

# 保存结果到新的Excel文件
output_file = os.path.join(current_dir, '敏感性和特异性汇总结果.xlsx')
results_df.to_excel(output_file, index=False)

print(f"\n所有文件处理完成，结果已保存到: {output_file}")
print(f"总共处理了 {len(results)} 个文件")




