"""
文本质量评估脚本
使用BLEU-2、Rouge-L、METEOR、DIST-2四个指标评估文本质量
"""

import re
import jieba
import numpy as np
from collections import Counter
import math


def read_case_summaries(file_path):
    """
    读取案例症状总结文件，返回案例ID和对应文本的字典      
    返回:
        dict: 案例ID到文本的映射字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式分割案例
    cases = {}
    pattern = r'id=(\d+)\s*\n([^id=]+)'
    matches = re.findall(pattern, content)
    
    for case_id, text in matches:
        cases[int(case_id)] = text.strip()
    
    return cases


def tokenize_chinese(text):
    """
    使用jieba对中文文本进行分词
    """
    return list(jieba.cut(text))


def calculate_bleu_2(candidate, references):
    """
    计算BLEU-2分数
    
    参数:
        candidate (list): 候选文本的分词列表
        references (list): 参考文本的分词列表
        
    返回:
        float: BLEU-2分数
    """
    # 计算2-gram精度
    candidate_2grams = [' '.join(candidate[i:i+2]) for i in range(len(candidate)-1)]
    
    max_ref_length = max(len(ref) for ref in references)
    candidate_length = len(candidate)
    
    # 长度惩罚
    if candidate_length > max_ref_length:
        bp = 1
    else:
        bp = math.exp(1 - max_ref_length / candidate_length)
    
    # 计算2-gram匹配数
    matches = 0
    total = len(candidate_2grams)
    
    for gram in candidate_2grams:
        max_ref_count = 0
        for ref in references:
            ref_2grams = [' '.join(ref[i:i+2]) for i in range(len(ref)-1)]
            count = ref_2grams.count(gram)
            if count > max_ref_count:
                max_ref_count = count
        
        if max_ref_count > 0:
            matches += 1
    
    precision = matches / total if total > 0 else 0
    bleu_score = bp * precision
    
    return bleu_score


def calculate_rouge_l(candidate, reference):
    """
    计算ROUGE-L分数（基于最长公共子序列）
    
    参数:
        candidate (list): 候选文本的分词列表
        reference (list): 参考文本的分词列表
        
    返回:
        dict: 包含召回率、精确率和F1分数的字典
    """
    def longest_common_subsequence(X, Y):
        """计算两个序列的最长公共子序列长度"""
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_length = longest_common_subsequence(candidate, reference)
    
    if len(reference) == 0 or len(candidate) == 0:
        return {'recall': 0, 'precision': 0, 'f1': 0}
    
    recall = lcs_length / len(reference)
    precision = lcs_length / len(candidate)
    
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    
    return {'recall': recall, 'precision': precision, 'f1': f1}


def calculate_meteor(candidate, reference, alpha=0.9, beta=3, gamma=0.5):
    """
    计算METEOR分数
    
    参数:
        candidate (list): 候选文本的分词列表
        reference (list): 参考文本的分词列表
        alpha, beta, gamma: METEOR参数
        
    返回:
        float: METEOR分数
    """
    # 计算精确匹配
    matches = 0
    candidate_matched = [False] * len(candidate)
    reference_matched = [False] * len(reference)
    
    for i, c_word in enumerate(candidate):
        for j, r_word in enumerate(reference):
            if not reference_matched[j] and c_word == r_word:
                matches += 1
                candidate_matched[i] = True
                reference_matched[j] = True
                break
    
    if matches == 0:
        return 0
    
    # 计算精度和召回率
    precision = matches / len(candidate)
    recall = matches / len(reference)
    
    if precision + recall == 0:
        f_mean = 0
    else:
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # 计算分块惩罚
    chunks = 0
    in_chunk = False
    
    for i in range(len(candidate)):
        if candidate_matched[i]:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
    
    penalty = gamma * (chunks / matches) ** beta
    
    meteor_score = f_mean * (1 - penalty)
    
    return meteor_score


def calculate_distinct_2(candidate):
    """
    计算DIST-2分数（多样性指标）
    
    参数:
        candidate (list): 候选文本的分词列表
        
    返回:
        float: DIST-2分数
    """
    if len(candidate) < 2:
        return 0
    
    # 计算2-gram
    bigrams = [' '.join(candidate[i:i+2]) for i in range(len(candidate)-1)]
    
    if len(bigrams) == 0:
        return 0
    
    # 计算不同的2-gram数量
    distinct_bigrams = len(set(bigrams))
    
    return distinct_bigrams / len(bigrams)


def evaluate_text_quality(generated_file, reference_file):
    """
    主评估函数
    
    参数:
        generated_file (str): 生成文本文件路径
        reference_file (str): 参考文本文件路径
        
    返回:
        dict: 包含所有评估结果的字典
    """
    # 读取文件
    print("正在读取文件...")
    generated_cases = read_case_summaries(generated_file)
    reference_cases = read_case_summaries(reference_file)
    
    print(f"生成文本案例数: {len(generated_cases)}")
    print(f"参考文本案例数: {len(reference_cases)}")
    
    # 找到共同的案例ID
    common_ids = set(generated_cases.keys()) & set(reference_cases.keys())
    print(f"共同案例数: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("错误: 没有找到共同的案例ID")
        return None
    
    # 初始化结果存储
    results = {
        'bleu_2': [],
        'rouge_l_recall': [],
        'rouge_l_precision': [],
        'rouge_l_f1': [],
        'meteor': [],
        'distinct_2': []
    }
    
    # 对每个案例进行评估
    print("\n开始评估...")
    for case_id in sorted(common_ids):
        print(f"评估案例 {case_id}...")
        
        # 分词
        candidate_tokens = tokenize_chinese(generated_cases[case_id])
        reference_tokens = tokenize_chinese(reference_cases[case_id])
        
        # 计算各项指标
        bleu_score = calculate_bleu_2(candidate_tokens, [reference_tokens])
        rouge_scores = calculate_rouge_l(candidate_tokens, reference_tokens)
        meteor_score = calculate_meteor(candidate_tokens, reference_tokens)
        distinct_score = calculate_distinct_2(candidate_tokens)
        
        # 存储结果
        results['bleu_2'].append(bleu_score)
        results['rouge_l_recall'].append(rouge_scores['recall'])
        results['rouge_l_precision'].append(rouge_scores['precision'])
        results['rouge_l_f1'].append(rouge_scores['f1'])
        results['meteor'].append(meteor_score)
        results['distinct_2'].append(distinct_score)
    
    # 计算平均值
    avg_results = {}
    for metric, values in results.items():
        avg_results[metric] = np.mean(values)
    
    return {
        'detailed_results': results,
        'average_results': avg_results,
        'case_count': len(common_ids)
    }


def print_results(evaluation_results):
    """
    打印评估结果
    
    参数:
        evaluation_results (dict): 评估结果
    """
    if evaluation_results is None:
        return
    
    avg_results = evaluation_results['average_results']
    case_count = evaluation_results['case_count']
    
    print("\n" + "="*60)
    print("文本质量评估结果")
    print(f"评估案例数: {case_count}")
    
    print("平均指标分数:")
    print(f"BLEU-2: {avg_results['bleu_2']:.4f}")
    print(f"ROUGE-L 召回率: {avg_results['rouge_l_recall']:.4f}")
    print(f"ROUGE-L 精确率: {avg_results['rouge_l_precision']:.4f}")
    print(f"ROUGE-L F1分数: {avg_results['rouge_l_f1']:.4f}")
    print(f"METEOR: {avg_results['meteor']:.4f}")
    print(f"DIST-2: {avg_results['distinct_2']:.4f}")
    
    # 保存详细结果到文件
    save_detailed_results(evaluation_results)


def save_detailed_results(evaluation_results):
    """
    保存详细评估结果到文件
    
    参数:
        evaluation_results (dict): 评估结果
    """
    with open('详细评估结果_标准病例无RAG_Qwen3235BA.txt', 'w', encoding='utf-8') as f:
        f.write("文本质量详细评估结果\n")
        f.write("="*50 + "\n\n")
        
        # 平均结果
        avg_results = evaluation_results['average_results']
        f.write("平均指标分数:\n")
        f.write(f"BLEU-2: {avg_results['bleu_2']:.4f}\n")
        f.write(f"ROUGE-L 召回率: {avg_results['rouge_l_recall']:.4f}\n")
        f.write(f"ROUGE-L 精确率: {avg_results['rouge_l_precision']:.4f}\n")
        f.write(f"ROUGE-L F1分数: {avg_results['rouge_l_f1']:.4f}\n")
        f.write(f"METEOR: {avg_results['meteor']:.4f}\n")
        f.write(f"DIST-2: {avg_results['distinct_2']:.4f}\n\n")
        
        # 详细结果
        detailed_results = evaluation_results['detailed_results']
        f.write("各案例详细分数:\n")
        f.write("案例ID\tBLEU-2\tROUGE-L F1\tMETEOR\tDIST-2\n")
        
        for i, case_id in enumerate(range(1, evaluation_results['case_count'] + 1)):
            f.write(f"{case_id}\t")
            f.write(f"{detailed_results['bleu_2'][i]:.4f}\t")
            f.write(f"{detailed_results['rouge_l_f1'][i]:.4f}\t")
            f.write(f"{detailed_results['meteor'][i]:.4f}\t")
            f.write(f"{detailed_results['distinct_2'][i]:.4f}\n")
    
    print("\n详细结果已保存")


if __name__ == "__main__":
    # 文件路径
    generated_file = "标准病例症状总结结果无RAG_Qwen3235BA.txt"
    reference_file = "标准病例症状总结.txt"
    
    try:
        # 执行评估
        results = evaluate_text_quality(generated_file, reference_file)
        
        # 打印结果
        print_results(results)
        
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请确保文件存在于当前目录中")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()