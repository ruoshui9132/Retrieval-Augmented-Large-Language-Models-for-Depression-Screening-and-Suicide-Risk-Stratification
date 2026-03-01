import pandas as pd
from sklearn.metrics import classification_report
import warnings
import os

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
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建一个空字典来存储所有文件的结果
    all_results = {}
    
    try:
        # 遍历目录下所有的Excel文件
        for filename in os.listdir(current_dir):
            if filename.endswith('.xlsx') and filename != '四分类指标计算结果.xlsx':
                file_path = os.path.join(current_dir, filename)
                print(f"\n处理文件: {filename}")
                
                # 计算指标
                metrics = calculate_metrics(file_path)
                
                # 输出结果
                print(f"\n===== {filename} 计算结果 =====")
                for task, scores in metrics.items():
                    print(f"\n{task}四分类任务指标:")
                    print(f"加权精确率: {scores['加权精确率']:.4f}")
                    print(f"加权召回率: {scores['加权召回率']:.4f}")
                    print(f"加权F1值: {scores['加权F1值']:.4f}")
                
                # 保存结果到总字典中，使用文件名作为键
                all_results[filename] = metrics
        
        # 将所有结果整理成一个DataFrame
        # 创建一个空的DataFrame用于存储所有结果
        result_rows = []
        
        # 遍历每个文件的结果
        for filename, metrics in all_results.items():
            # 提取抑郁症指标
            depression_metrics = metrics['抑郁症']
            result_rows.append({
                '文件名': filename,
                '类别': '抑郁症',
                '加权精确率': depression_metrics['加权精确率'],
                '加权召回率': depression_metrics['加权召回率'],
                '加权F1值': depression_metrics['加权F1值']
            })
            
            # 提取自杀风险指标
            suicide_metrics = metrics['自杀风险']
            result_rows.append({
                '文件名': filename,
                '类别': '自杀风险',
                '加权精确率': suicide_metrics['加权精确率'],
                '加权召回率': suicide_metrics['加权召回率'],
                '加权F1值': suicide_metrics['加权F1值']
            })
        
        # 创建DataFrame
        result_df = pd.DataFrame(result_rows)
        
        # 将结果保存到Excel文件
        result_file = os.path.join(current_dir, '标准病例四分类指标计算结果.xlsx')
        result_df.to_excel(result_file, index=False)
        print(f"\n所有结果已保存到文件: {result_file}")
        
    except Exception as e:
        print(f"程序执行失败: {str(e)}")

if __name__ == "__main__":
    main()