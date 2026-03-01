import pandas as pd
import os

def calculate_file_accuracy(file_path):
    """
    计算单个Excel文件的抑郁症和自杀风险预测正确率
    
    Args:
        file_path: Excel文件路径
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 确保必要的列存在
    required_columns = ['有无抑郁症', '抑郁症', '有无自杀风险', '自杀风险']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"文件 {os.path.basename(file_path)} 中缺少必要的列: {col}")
    
    # 计算抑郁症的正确率
    # '抑郁症'列是标准答案，'有无抑郁症'列是待评估的预测结果
    depression_correct = (df['有无抑郁症'] == df['抑郁症']).sum()
    depression_total = len(df)
    depression_accuracy = depression_correct / depression_total
    
    # 计算自杀风险的正确率
    # '自杀风险'列是标准答案（0代表无，非0代表有）
    # 需要将标准答案转换为文本分类('无'/'有')以便与'有无自杀风险'列比较
    df['自杀风险_text'] = df['自杀风险'].apply(lambda x: '有' if x != 0 else '无')
    suicide_correct = (df['有无自杀风险'] == df['自杀风险_text']).sum()
    suicide_total = len(df)
    suicide_accuracy = suicide_correct / suicide_total
    
    # 输出结果
    print(f"文件: {os.path.basename(file_path)}")
    print(f"  抑郁症预测正确率: {depression_accuracy*100:.4f} ({depression_correct}/{depression_total})")
    print(f"  自杀风险预测正确率: {suicide_accuracy*100:.4f} ({suicide_correct}/{suicide_total})")


def calculate_all_files_accuracy():
    """
    遍历当前目录下所有Excel文件并计算每个文件的诊断正确率
    """
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化统计变量
    total_files = 0
    
    # 遍历目录下所有文件
    for filename in os.listdir(current_dir):
        # 检查文件是否为Excel文件
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            total_files += 1
            file_path = os.path.join(current_dir, filename)
            try:
                # 计算单个文件的准确率
                calculate_file_accuracy(file_path)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                print()
    
    # 输出处理的文件总数
    if total_files > 0:
        print(f"共处理了 {total_files} 个Excel文件")
    else:
        print("未找到Excel文件")

if __name__ == "__main__":
    calculate_all_files_accuracy()