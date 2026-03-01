import pandas as pd

# 读取Excel文件
df = pd.read_excel('BaichuanM2四分类任务_有RAG_第3次运行.xlsx')

# 计算抑郁症敏感性
# 抑郁症敏感性 = (抑郁症（标准）列不为'无'且和抑郁症列值一样的案例数) / (抑郁症列不为'无'的案例数)
depression_true_positive = len(df[(df['抑郁症（标准）'] != '无') & (df['抑郁症'] == df['抑郁症（标准）'])])
depression_positive = len(df[df['抑郁症（标准）'] != '无'])
depression_sensitivity = depression_true_positive / depression_positive if depression_positive > 0 else 0

# 计算抑郁症特异性
# 抑郁症特异性 = (抑郁症列为'无'且抑郁症（标准）列也为'无'的案例数) / (抑郁症列值为'无'的案例数)
depression_true_negative = len(df[(df['抑郁症'] == '无') & (df['抑郁症（标准）'] == '无')])
depression_negative = len(df[df['抑郁症（标准）'] == '无'])
depression_specificity = depression_true_negative / depression_negative if depression_negative > 0 else 0

# 计算自杀风险敏感性
# 自杀风险敏感性 = (自杀风险（标准）列不为'0'且和自杀风险列值一样的案例数) / (自杀风险列不为'0'的案例数)
suicide_true_positive = len(df[(df['自杀风险（标准）'] != 0) & (df['自杀风险'] == df['自杀风险（标准）'])])
suicide_positive = len(df[df['自杀风险（标准）'] != 0])
suicide_sensitivity = suicide_true_positive / suicide_positive if suicide_positive > 0 else 0

# 计算自杀风险特异性
# 自杀风险特异性 = (自杀风险列为'0'且自杀风险（标准）列也为'0'的案例数) / (自杀风险列为'0'的案例数)
suicide_true_negative = len(df[(df['自杀风险'] == 0) & (df['自杀风险（标准）'] == 0)])
suicide_negative = len(df[df['自杀风险（标准）'] == 0])
suicide_specificity = suicide_true_negative / suicide_negative if suicide_negative > 0 else 0

print("\n抑郁症统计:")
print(f"抑郁症（标准）列不为'无'的案例总数: {depression_positive}")
print(f"抑郁症（标准）列不为'无'且和抑郁症列值一样的案例数: {depression_true_positive}")
print(f"抑郁症敏感性:{depression_sensitivity*100:.2f}%")

print(f"\n抑郁症（标准）列值为'无'的案例总数: {depression_negative}")
print(f"抑郁症列为'无'且抑郁症（标准）列也为'无'的案例数: {depression_true_negative}")
print(f"抑郁症特异性: {depression_specificity*100:.2f}%")

print("\n自杀风险统计:")
print(f"自杀风险（标准）列不为'0'的案例总数: {suicide_positive}")
print(f"自杀风险（标准）列不为'0'且和自杀风险列值一样的案例数: {suicide_true_positive}")
print(f"自杀风险敏感性: {suicide_sensitivity*100:.2f}%")

print(f"\n自杀风险（标准）列为'0'的案例总数: {suicide_negative}")
print(f"自杀风险列为'0'且自杀风险（标准）列也为'0'的案例数: {suicide_true_negative}")
print(f"自杀风险特异性: {suicide_specificity*100:.2f}%")


print("\n" + "="*50)
print("抑郁症分类详细统计:")
print("="*50)
for depression_type in df['抑郁症（标准）'].unique():
    if depression_type != '无':
        count = len(df[df['抑郁症（标准）'] == depression_type])
        correct_count = len(df[(df['抑郁症（标准）'] == depression_type) & (df['抑郁症'] == depression_type)])
        accuracy = correct_count / count if count > 0 else 0
        print(f"{depression_type}类: 总数={count}, 正确数={correct_count}, 准确率={accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n自杀风险分类详细统计:")
print("="*50)
for suicide_type in df['自杀风险（标准）'].unique():
    if suicide_type != 0:
        count = len(df[df['自杀风险（标准）'] == suicide_type])
        correct_count = len(df[(df['自杀风险（标准）'] == suicide_type) & (df['自杀风险'] == suicide_type)])
        accuracy = correct_count / count if count > 0 else 0
        print(f"{suicide_type}类: 总数={count}, 正确数={correct_count}, 准确率={accuracy:.4f} ({accuracy*100:.2f}%)")