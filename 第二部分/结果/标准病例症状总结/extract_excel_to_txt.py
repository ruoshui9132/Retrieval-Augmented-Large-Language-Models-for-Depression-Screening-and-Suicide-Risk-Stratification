import pandas as pd
import os


def extract_id_and_summary(excel_file, output_file):
    """   
    参数:
        excel_file (str): Excel文件路径
        output_file (str): 输出的txt文件路径
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        
        # 确保必要的列存在
        # 获取第一列(Id)和第四列(症状总结)
        # 注意：pandas的列索引从0开始
        ids = df.iloc[:, 0]  # 第一列
        summaries = df.iloc[:, 3]  # 第四列
        
        # 打开输出文件准备写入
        with open(output_file, 'w', encoding='utf-8') as f:
            # 遍历每一行数据
            for idx, (id_value, summary) in enumerate(zip(ids, summaries)):
                # 写入id
                f.write(f"id={id_value}\n")
                # 写入症状总结
                # 确保summary是字符串类型，并且去除可能存在的换行符
                if pd.isna(summary):
                    summary_text = ""
                else:
                    summary_text = str(summary).replace('\n', ' ').strip()
                f.write(f"{summary_text}\n\n")
        
        print(f"数据已成功提取到{output_file}")
        print(f"共处理了{len(ids)}条记录")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Excel文件路径
    excel_file = "抑郁症标准病例编号和症状总结.xlsx"
    # 输出的txt文件路径
    output_file = "标准病例症状总结_extracted.txt"
    
    # 检查Excel文件是否存在
    if not os.path.exists(excel_file):
        print(f"错误: 找不到Excel文件 '{excel_file}'")
        print("请确保文件存在于当前目录中")
    else:
        # 执行提取操作
        extract_id_and_summary(excel_file, output_file)