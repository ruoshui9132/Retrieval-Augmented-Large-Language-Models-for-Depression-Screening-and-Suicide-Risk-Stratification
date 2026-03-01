import pandas as pd

def calculate_accuracy():
    # 读取Excel文件
    file_path = '千问四分类任务_有RAG.xlsx'
    df = pd.read_excel(file_path)
    
    # 确保必要的列存在
    required_columns = ['抑郁症', '抑郁症（标准）', '自杀风险（标准）', '自杀风险']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"文件中缺少必要的列: {col}")
    
    # 计算抑郁症的正确率
    # '抑郁症'列是标准答案，'抑郁症（标准）'列是待评估的预测结果
    depression_correct = (df['抑郁症'] == df['抑郁症（标准）']).sum()
    depression_total = len(df)
    depression_accuracy = depression_correct / depression_total
    
    # 计算自杀风险的正确率
    suicide_correct = (df['自杀风险'] == df['自杀风险（标准）']).sum()
    suicide_total = len(df)
    suicide_accuracy = suicide_correct / suicide_total
    
    # 输出结果
    print(f"抑郁症预测正确率: {depression_accuracy:.4f} ({depression_correct}/{depression_total})")
    print(f"自杀风险预测正确率: {suicide_accuracy:.4f} ({suicide_correct}/{suicide_total})")

if __name__ == "__main__":
    calculate_accuracy()