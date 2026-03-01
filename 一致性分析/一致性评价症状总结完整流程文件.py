"""
一致性评价十次运行分析脚本
对每个案例对话重复运行十次症状总结生成，评估相似性并统计分析
"""

import os
import re
import time
import openai
import pandas as pd
import numpy as np
import jieba
import math
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import scipy.stats as stats

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key="sk-FCLbw1bLNBABZksmri1IlQ",
    base_url="https://llmapi.paratera.com"
)

# 读取案例文档
# 注释：从指定文件路径读取案例对话，返回案例ID和对话内容的字典
# 使用正则表达式匹配案例格式，确保正确解析每个案例
# 返回格式：{案例ID: 对话内容}
def read_cases(file_path: str) -> Dict[int, str]:
    """读取案例对话文件，返回案例ID和对话内容的字典"""
    cases = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 正则表达式匹配案例，每个案例以id=X开头
        case_pattern = re.compile(r'id=(\d+)\n(.*?)(?=\nid=\d+|$)', re.DOTALL)
        matches = case_pattern.findall(content)
        
        for case_id, case_content in matches:
            cases[int(case_id)] = case_content.strip()
    
    return cases

# 读取知识库
# 注释：读取知识库文件并解析为结构化数据
# 提取主题、摘选、来源等信息，用于后续的知识检索
# 返回格式：知识块字典列表
def read_knowledge_base(file_path: str) -> List[Dict[str, str]]:
    """读取知识库文件，解析为知识块列表"""
    knowledge_base = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        pattern = re.compile(
            r'主题：(.*?)\n'
            r'摘选：(.*?)\n'
            r'来源：(.*?)\n'
            r'页码：(.*?)\n'
            r'收录日期：(.*?)(?=\n主题：|$)', 
            re.DOTALL
        )
        
        matches = pattern.findall(content)
        for match in matches:
            knowledge_base.append({
                "主题": match[0].strip(),
                "摘选": match[1].strip(),
                "来源": match[2].strip(),
                "页码": match[3].strip(),
                "收录日期": match[4].strip(),
                # 预处理：合并主题和摘选用于检索
                "combined_text": f"{match[0].strip()} {match[1].strip()}"
            })
    
    return knowledge_base

# 构建知识库检索器
# 注释：基于TF-IDF的知识检索器，用于检索与症状总结相关的专业知识
# 使用症状总结和临床评估两个任务需求作为查询向量
class TaskOnlyKnowledgeRetriever:
    def __init__(self, knowledge_base: List[Dict[str, str]]):
        self.knowledge_base = knowledge_base
        self.texts = [item["combined_text"] for item in knowledge_base]
        
        # 定义核心任务需求 - 调整为与症状总结相关
        self.task_queries = {
            "symptom_summary": "如何撰写抑郁症相关症状总结",
            "clinical_evaluation": "临床心理评估的关键要素和表述方式"
        }
        
        # 初始化TF-IDF向量器
        self.vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),
            max_features=10000
        )
        # 拟合所有知识库文本和任务查询
        all_texts = self.texts + list(self.task_queries.values())
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # 提取任务查询的向量（最后两个向量）
        self.task_vectors = {
            "symptom_summary": self.tfidf_matrix[-2:-1],  # 症状总结任务向量
            "clinical_evaluation": self.tfidf_matrix[-1:]   # 临床评估任务向量
        }
        
        # 知识库文本向量（排除任务查询）
        self.kb_matrix = self.tfidf_matrix[:-2]
    
    def retrieve(self, top_k: int = 3, 
                symptom_weight: float = 0.5,
                evaluation_weight: float = 0.5) -> List[Dict[str, str]]:
        """
        仅基于核心任务需求检索相关知识块
        权重参数控制两个任务的影响程度，总和应为1
        """
        # 计算与症状总结任务的相似度
        symptom_similarities = cosine_similarity(
            self.task_vectors["symptom_summary"], self.kb_matrix
        ).flatten()
        
        # 计算与临床评估任务的相似度
        evaluation_similarities = cosine_similarity(
            self.task_vectors["clinical_evaluation"], self.kb_matrix
        ).flatten()
        
        # 综合相似度：仅基于任务需求的加权求和
        combined_similarities = (
            symptom_weight * symptom_similarities +
            evaluation_weight * evaluation_similarities
        )
        
        top_indices = combined_similarities.argsort()[-top_k:][::-1]
        
        # 返回最相关的知识块，附带相似度分数
        results = []
        for i in top_indices:
            if combined_similarities[i] > 0.01:  # 过滤低相似度
                result = self.knowledge_base[i].copy()
                result["similarity_score"] = float(combined_similarities[i])
                results.append(result)
        
        return results

# 构建提示词
# 注释：构建用于LLM生成症状总结的提示词，包含检索到的相关知识
# 遵循标准的症状总结格式要求，确保输出一致性
def build_prompt(case_content: str, relevant_knowledge: List[Dict[str, str]]) -> str:
    """构建用于LLM生成症状总结的提示词，包含检索到的相关知识"""
    # 格式化相关知识
    knowledge_text = ""
    if relevant_knowledge:
        knowledge_text = "参考以下专业知识（按相关性排序）：\n"
        for i, knowledge in enumerate(relevant_knowledge, 1):
            knowledge_text += f"{i}. 主题：{knowledge['主题']}\n"
            knowledge_text += f"   内容：{knowledge['摘选']}\n"
            knowledge_text += f"   来源：{knowledge['来源']}（{knowledge['收录日期']}）\n\n"
    
    prompt = f"""
    请分析以下医患沟通对话，生成一份症状总结。
    一下是生成症状总结的要求：
主题：抑郁症病例症状总结书写标准
摘选：【症状总结要求】：
【症状总结输出格式】：
来访者（[年龄][性别]） [症状时长][核心症状①][核心症状②][核心症状③]。伴[其他症状①]、[其他症状②]、[睡眠障碍类型]、[食欲变化描述]。存在[自杀风险描述]。功能损害摘要。量表分数。处置建议。
【症状映射规则】：
█ 核心症状：
① 心境低落 → 标注具体表现（如晨间哭泣/持续低落）
② 兴趣/愉快感丧失 → 注明放弃的活动（如拒春游/拒游戏）
③ 精力减退 → 引用疲劳描述（如乏力/睡多仍困）
█ 其他症状从以下选择2项优先项：
• 注意力降低 → 注明场景（如上课走神/读三遍）
• 自我评价低 → 引用自责原话（如"失败者"/评分8分）
• 无价值感 → 标注频次（如反复想离婚）
• 前途悲观 → 引用原话（如"人生完了"）
• 自杀/自伤 → 分级标注：
  - 0=无 → 省略此项
  - 1=意念 → "自杀意念（原话）"
  - 2=计划 → "自杀计划（方法）"
  - 3=行为 → "自伤行为（方式）"
• 睡眠障碍 → 注明类型（早醒/难入睡/多梦/嗜睡）
• 食欲下降 → 标注体重变化/暴食厌食
【强制要求】：
1. 所有症状必须带括号补充医患对话原句关键词（不超过10字）
2. 时长/体重/评分必须保留数字（如3个月/2kg/PHQ-11）
3. 功能损害用10字内概括（如成绩下滑/社交回避）
4. 禁用分点/列表，整段在250-300字符之间


     下面是一个症状总结示例：
     来访者（37岁女性）近半年情绪低落（罩玻璃罐），兴趣丧失（拒拆化妆品），精力减退（醒后更累）。伴自我评价低（扇耳光）、睡眠障碍（多梦易醒）、自杀意念（"活着没意思"）。工作请假增多。PHQ-14分。建议立即精神科急诊就诊，严格防范自杀风险，启动危机干预。

    仿照这个示例，回答中请只生成症状总结内容，不要添加任何其他内容，这一点非常重要。

    
    医患沟通对话内容：
    {case_content}
    """
    return prompt.strip()

# 调用LLM API生成症状总结
# 注释：调用OpenAI API生成症状总结，包含重试机制和错误处理
# 设置temperature=0.3降低随机性，确保结果稳定性
def generate_summary(case_content: str, retriever: TaskOnlyKnowledgeRetriever, model_name: str = "Qwen3-235B-A22B-Instruct-2507", max_retries: int = 3) -> str:
    """调用LLM API分析单个案例，生成症状总结"""
    # 检索相关知识
    relevant_knowledge = retriever.retrieve()
    
    # 构建带知识的提示词
    prompt = build_prompt(case_content, relevant_knowledge)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一位经验丰富的心理医生，擅长分析病例对话，能根据医患沟通内容生成专业简洁的症状总结。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低随机性，使结果更稳定
                max_tokens=15000     # 限制输出长度
            )
            
            summary = response.choices[0].message.content.strip()
            print(f"使用模型 {model_name} 生成总结: {summary[:100]}...")  # 打印前100个字符
            return summary
        
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2** attempt)  # 指数退避
                continue
    
    # 如果多次尝试失败，返回错误信息
    return f"使用模型 {model_name} 生成总结失败"

# 文本相似性评估函数
# 注释：以下函数用于计算四个文本相似性指标
# 包括BLEU-2、ROUGE-L、METEOR和DIST-2

def tokenize_chinese(text):
    """使用jieba对中文文本进行分词"""
    return list(jieba.cut(text))

def calculate_bleu_2(candidate, references):
    """计算BLEU-2分数"""
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
    """计算ROUGE-L分数（基于最长公共子序列）"""
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
    """计算METEOR分数"""
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
    """计算DIST-2分数（多样性指标）"""
    if len(candidate) < 2:
        return 0
    
    # 计算2-gram
    bigrams = [' '.join(candidate[i:i+2]) for i in range(len(candidate)-1)]
    
    if len(bigrams) == 0:
        return 0
    
    # 计算不同的2-gram数量
    distinct_bigrams = len(set(bigrams))
    
    return distinct_bigrams / len(bigrams)

# 评估单个案例的相似性
# 注释：对每个案例的10次运行结果进行相似性评估
# 以第一次运行结果为参考，计算与其他9次的相似性指标
def evaluate_case_similarity(case_results: List[str]) -> Dict[str, List[float]]:
    """
    评估单个案例10次运行结果的相似性
    以第一次运行结果为参考，计算与其他9次的相似性
    
    参数:
        case_results: 单个案例的10次运行结果列表
        
    返回:
        包含四个指标结果的字典
    """
    if len(case_results) != 10:
        raise ValueError("每个案例必须有10次运行结果")
    
    # 第一次运行结果作为参考文本
    reference_text = case_results[0]
    reference_tokens = tokenize_chinese(reference_text)
    
    # 初始化结果存储
    results = {
        'bleu_2': [],
        'rouge_l_f1': [],
        'meteor': [],
        'distinct_2': []
    }
    
    # 对其他9次运行结果进行评估
    for i in range(1, 10):
        candidate_text = case_results[i]
        candidate_tokens = tokenize_chinese(candidate_text)
        
        # 计算BLEU-2
        bleu_score = calculate_bleu_2(candidate_tokens, [reference_tokens])
        results['bleu_2'].append(bleu_score)
        
        # 计算ROUGE-L F1分数
        rouge_scores = calculate_rouge_l(candidate_tokens, reference_tokens)
        results['rouge_l_f1'].append(rouge_scores['f1'])
        
        # 计算METEOR
        meteor_score = calculate_meteor(candidate_tokens, reference_tokens)
        results['meteor'].append(meteor_score)
        
        # 计算DIST-2
        distinct_score = calculate_distinct_2(candidate_tokens)
        results['distinct_2'].append(distinct_score)
    
    return results

# 统计分析函数
# 注释：计算30个案例的平均值、标准差和95%置信区间
def calculate_statistics(all_case_results: Dict[int, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """
    计算所有案例的统计指标
    
    参数:
        all_case_results: 所有案例的相似性结果字典
        
    返回:
        包含平均值、标准差和置信区间的字典
    """
    # 初始化指标数据存储
    metrics_data = {
        'bleu_2': [],
        'rouge_l_f1': [],
        'meteor': [],
        'distinct_2': []
    }
    
    # 收集所有案例的相似性数据
    for case_id, case_results in all_case_results.items():
        for metric, values in case_results.items():
            metrics_data[metric].extend(values)
    
    # 计算统计指标
    statistics = {}
    for metric, values in metrics_data.items():
        if len(values) == 0:
            statistics[metric] = {
                'mean': 0,
                'std': 0,
                'ci_lower': 0,
                'ci_upper': 0
            }
            continue
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # 样本标准差
        n = len(values)
        
        # 计算95%置信区间
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
            ci_lower, ci_upper = ci
        else:
            ci_lower, ci_upper = mean, mean
        
        statistics[metric] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return statistics

# 主函数
# 注释：整合所有功能，执行十次运行、相似性评估和统计分析
def main():
    """主函数：执行十次运行、相似性评估和统计分析"""
    print("=== 一致性评价十次运行分析开始 ===")
    
    # 配置参数
    input_file = "一致性评价对话.txt"       # 案例对话文件
    knowledge_file = "知识库.txt"           # 知识库文件
    output_excel = "一致性评价十次症状总结运行结果.xlsx"  # 输出Excel文件
    num_runs = 10                          # 每个案例运行次数
    api_delay = 1.0                        # API调用延迟
    
    # 读取案例和知识库
    print("1. 读取案例对话和知识库...")
    cases = read_cases(input_file)
    knowledge_base = read_knowledge_base(knowledge_file)
    retriever = TaskOnlyKnowledgeRetriever(knowledge_base)
    
    print(f"共读取到 {len(cases)} 个案例")
    print(f"知识库包含 {len(knowledge_base)} 个知识块")
    
    # 存储所有案例的所有运行结果
    # 结构：all_results[运行次数][案例ID] = 症状总结
    all_results = {}
    
    # 执行10次完整流程：依次处理所有案例
    print("\n2. 开始十次完整流程运行...")
    for run_num in range(num_runs):
        print(f"\n=== 第{run_num+1}次完整流程运行开始 ===")
        run_results = {}
        
        # 依次处理所有案例
        for case_id, case_content in sorted(cases.items()):
            print(f"处理案例 {case_id}...")
            summary = generate_summary(case_content, retriever)
            run_results[case_id] = summary
            
            # 添加延迟避免API限制
            time.sleep(api_delay)
        
        all_results[run_num + 1] = run_results
        print(f"=== 第{run_num+1}次完整流程运行完成 ===")
        
        # 在运行之间添加稍长延迟
        if run_num < num_runs - 1:
            time.sleep(2.0)
    
    # 保存所有运行结果到Excel
    print("\n3. 保存结果到Excel文件...")
    with pd.ExcelWriter(output_excel) as writer:
        # 创建数据框
        data_rows = []
        for run_num, run_results in all_results.items():
            for case_id, summary in run_results.items():
                data_rows.append({
                    '运行次数': run_num,
                    '案例ID': case_id,
                    '症状总结': summary
                })
        
        df = pd.DataFrame(data_rows)
        df.to_excel(writer, sheet_name='所有运行结果', index=False)
    
    print(f"结果已保存到 {output_excel}")
    
    # 重新组织数据结构：按案例ID分组
    # 将 all_results[运行次数][案例ID] 转换为 case_results[案例ID][运行次数]
    case_results = {}
    for case_id in sorted(cases.keys()):
        case_results[case_id] = []
        for run_num in range(1, num_runs + 1):
            case_results[case_id].append(all_results[run_num][case_id])
    
    # 评估相似性
    print("\n4. 开始相似性评估...")
    similarity_results = {}
    
    for case_id, results in case_results.items():
        print(f"评估案例 {case_id} 的相似性...")
        case_similarity = evaluate_case_similarity(results)
        similarity_results[case_id] = case_similarity
    
    # 计算统计指标
    print("\n5. 计算统计指标...")
    statistics = calculate_statistics(similarity_results)
    
    # 输出统计结果
    print("\n=== 统计结果 ===")
    print(f"评估案例总数: {len(similarity_results)}")
    print(f"每个案例评估次数: 9次")
    print(f"总评估次数: {len(similarity_results) * 9}")
    
    for metric, stats_info in statistics.items():
        print(f"\n{metric.upper()}:")
        print(f"  平均值: {stats_info['mean']:.4f}")
        print(f"  标准差: {stats_info['std']:.4f}")
        print(f"  95%置信区间: [{stats_info['ci_lower']:.4f}, {stats_info['ci_upper']:.4f}]")
    
    # 保存统计结果到Excel
    print("\n6. 保存统计结果...")
    with pd.ExcelWriter(output_excel, mode='a', if_sheet_exists='replace') as writer:
        # 相似性详细结果
        similarity_data = []
        for case_id, case_similarity in similarity_results.items():
            for run_num in range(9):  # 9次比较
                similarity_data.append({
                    '案例ID': case_id,
                    '比较次数': run_num + 1,
                    'BLEU-2': case_similarity['bleu_2'][run_num],
                    'ROUGE-L F1': case_similarity['rouge_l_f1'][run_num],
                    'METEOR': case_similarity['meteor'][run_num],
                    'DIST-2': case_similarity['distinct_2'][run_num]
                })
        
        df_similarity = pd.DataFrame(similarity_data)
        df_similarity.to_excel(writer, sheet_name='相似性评估', index=False)
        
        # 统计汇总结果
        stats_data = []
        for metric, stats_info in statistics.items():
            stats_data.append({
                '指标': metric.upper(),
                '平均值': stats_info['mean'],
                '标准差': stats_info['std'],
                '置信区间下限': stats_info['ci_lower'],
                '置信区间上限': stats_info['ci_upper']
            })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='统计汇总', index=False)
    
    print(f"统计结果已保存到 {output_excel}")
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()