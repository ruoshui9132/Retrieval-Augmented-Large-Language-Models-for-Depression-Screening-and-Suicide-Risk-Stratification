import pandas as pd
from sklearn.metrics import classification_report
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

def calculate_metrics(file_path):
    """
    计算四分类任务的加权精确率、加权召回率、加权F1值
    
    参数:
    file_path: Excel文件路径
    
    返回:
    字典，包含抑郁症和自杀风险的各项指标
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"成功读取文件，数据维度: {df.shape}")
        
        # 检查是否包含所需的列
        required_columns = ['案例编号', '抑郁症', '自杀风险', '抑郁症（标准）', '自杀风险（标准）']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"文件中缺少列: {col}")
        
        print("数据列名检查通过")
        
        # 提取数据
        y_true_depression = df['抑郁症（标准）']
        y_pred_depression = df['抑郁症']
        y_true_suicide = df['自杀风险（标准）']
        y_pred_suicide = df['自杀风险']
        
        print("开始计算抑郁症四分类任务指标...")
        # 计算抑郁症四分类的指标
        report_depression = classification_report(y_true_depression, y_pred_depression, output_dict=True)
        
        print("开始计算自杀风险四分类任务指标...")
        # 计算自杀风险四分类的指标
        report_suicide = classification_report(y_true_suicide, y_pred_suicide, output_dict=True)
        
        # 提取加权指标
        metrics = {
            '抑郁症': {
                '加权精确率': report_depression['weighted avg']['precision'],
                '加权召回率': report_depression['weighted avg']['recall'],
                '加权F1值': report_depression['weighted avg']['f1-score']
            },
            '自杀风险': {
                '加权精确率': report_suicide['weighted avg']['precision'],
                '加权召回率': report_suicide['weighted avg']['recall'],
                '加权F1值': report_suicide['weighted avg']['f1-score']
            }
        }
        
        return metrics
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

def main():
    """
    主函数，用于执行指标计算并输出结果
    """
    # 文件路径 - 使用原始字符串格式避免转义问题
    file_path = r'd:\大三下学习\科研路\自己的方向\文章结果\第一部分\结果\好大夫四分类\BaichuanM2四分类任务_有RAG_第3次运行.xlsx'
    
    try:
        # 计算指标
        metrics = calculate_metrics(file_path)
        
        # 输出结果
        print("\n===== 计算结果 =====")
        for task, scores in metrics.items():
            print(f"\n{task}四分类任务指标:")
            print(f"加权精确率: {scores['加权精确率']:.4f}")
            print(f"加权召回率: {scores['加权召回率']:.4f}")
            print(f"加权F1值: {scores['加权F1值']:.4f}")
            
#         # 将结果保存到Excel文件
#         result_df = pd.DataFrame.from_dict(metrics, orient='index')
#         result_file = '四分类指标计算结果.xlsx'
#         result_df.to_excel(result_file)
#         print(f"\n结果已保存到文件: {result_file}")
        
    except Exception as e:
        print(f"程序执行失败: {str(e)}")

if __name__ == "__main__":
    main()