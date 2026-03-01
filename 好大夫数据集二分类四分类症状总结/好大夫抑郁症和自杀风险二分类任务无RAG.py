import os
import re
import time
import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key="sk-FCLbw1bLNBABZksmri1IlQ",
    base_url="https://llmapi.paratera.com"
)

# 读取案例文档
def read_cases(file_path: str) -> Dict[int, str]:
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

# 构建增强型知识库检索器（仅基于任务需求）
class TaskOnlyKnowledgeRetriever:
    def __init__(self, knowledge_base: List[Dict[str, str]]):
        self.knowledge_base = knowledge_base
        self.texts = [item["combined_text"] for item in knowledge_base]
        
        # 定义核心任务需求
        self.task_queries = {
            "depression": "根据DSM-5标准辨别患者是否为抑郁症",
            "suicide_risk": "评估自杀风险的指标和特征"
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
            "depression": self.tfidf_matrix[-2:-1],  # 抑郁症诊断任务向量
            "suicide_risk": self.tfidf_matrix[-1:]   # 自杀风险评估任务向量
        }
        
        # 知识库文本向量（排除任务查询）
        self.kb_matrix = self.tfidf_matrix[:-2]
    
    def retrieve(self, top_k: int = 3, 
                depression_weight: float = 0.5,
                suicide_risk_weight: float = 0.5) -> List[Dict[str, str]]:
        """
        仅基于核心任务需求检索相关知识块
        权重参数控制两个任务的影响程度，总和应为1
        """
        # 计算与抑郁症诊断任务的相似度
        depression_similarities = cosine_similarity(
            self.task_vectors["depression"], self.kb_matrix
        ).flatten()
        
        # 计算与自杀风险评估任务的相似度
        suicide_similarities = cosine_similarity(
            self.task_vectors["suicide_risk"], self.kb_matrix
        ).flatten()
        
        # 综合相似度：仅基于任务需求的加权求和
        combined_similarities = (
            depression_weight * depression_similarities +
            suicide_risk_weight * suicide_similarities
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
def build_prompt(case_content: str, relevant_knowledge: List[Dict[str, str]]) -> str:
    """构建用于LLM判断的提示词，包含检索到的相关知识"""
    # 格式化相关知识
    knowledge_text = ""
    if relevant_knowledge:
        knowledge_text = "参考以下专业知识（按相关性排序）：\n"
        for i, knowledge in enumerate(relevant_knowledge, 1):
            knowledge_text += f"{i}. 主题：{knowledge['主题']}\n"
            knowledge_text += f"   内容：{knowledge['摘选']}\n"
            knowledge_text += f"   来源：{knowledge['来源']}（{knowledge['收录日期']}）\n\n"
    
    prompt = f"""
    请分析以下医患沟通对话，完成两项判断：
    1. 判断患者是否有抑郁症。请仅回答"有"或"无"
    2. 患者是否有自杀风险？请仅回答"有"或"无"

    请将判断结果按照以下格式输出，判别结果请仅回答"有"或"无"，一定一定不要添加任何额外内容：
    抑郁症：[判别结果]
    自杀风险：[判别结果]
    
    医患沟通对话内容：
    {case_content}
    """
    return prompt.strip()

# 调用LLM API进行判断
def analyze_case(case_content: str, retriever: TaskOnlyKnowledgeRetriever, model_name: str = "Baichuan-M2", max_retries: int = 3) -> Tuple[str, str]:
    """调用LLM API分析单个案例，使用仅基于任务需求的RAG，返回抑郁症和自杀风险的判断结果"""
    # 检索相关知识（仅基于任务需求）
    relevant_knowledge = retriever.retrieve()
    
    # 构建带知识的提示词
    prompt = build_prompt(case_content, relevant_knowledge)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,  # 使用传入的模型名称
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低随机性，使结果更稳定
                max_tokens=10000     # 限制输出长度
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析结果
            depression_match = re.search(r'抑郁症：(有|无)', result)
            suicide_match = re.search(r'自杀风险：(有|无)', result)
            
            if depression_match and suicide_match:
                return depression_match.group(1), suicide_match.group(1)
            else:
                print(f"解析结果失败，结果格式不正确: {result}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 **attempt)  # 指数退避
                continue
    
    # 如果多次尝试失败，返回未知
    return "未知", "未知"

# 批量处理案例
def process_cases(cases: Dict[int, str], retriever: TaskOnlyKnowledgeRetriever, model_name: str = "Baichuan-M2", delay_seconds: float = 1.0) -> List[Dict]:
    """批量处理所有案例，返回包含判断结果的列表"""
    results = []
    total = len(cases)
    
    for i, (case_id, content) in enumerate(sorted(cases.items(), key=lambda x: x[0])):
        print(f"处理案例 {case_id} ({i+1}/{total})...")
        
        depression, suicide_risk = analyze_case(content, retriever, model_name)
        
        results.append({
            "案例编号": case_id,
            "有无抑郁症": depression,
            "有无自杀风险": suicide_risk
        })
        
        # 添加延迟，避免触发API速率限制
        if i < total - 1:
            time.sleep(delay_seconds)
    
    return results

# 保存结果到Excel
def save_results(results: List[Dict], output_file: str):
    """将判断结果保存到Excel文件"""
    df = pd.DataFrame(results)
    # 确保列的顺序正确
    df = df[["案例编号", "有无抑郁症", "有无自杀风险"]]
    df.to_excel(output_file, index=False)
    print(f"结果已保存到 {output_file}")

# 主函数
def main():
    # 配置参数
    input_file = "好大夫400例对话.txt"       # 包含案例的文本文件路径
    knowledge_file = "知识库.txt"             # 知识库文件路径
    api_delay = 1.0                           # API调用之间的延迟（秒）
    
    # 定义三个模型名称
    models = ["DeepSeek-R1"]
    #"Baichuan-M2", "Qwen3-235B-A22B-Instruct-2507",
    # 读取案例
    print(f"从 {input_file} 读取案例...")
    cases = read_cases(input_file)
    print(f"共读取到 {len(cases)} 个案例")
    
    # 读取知识库并创建仅基于任务需求的检索器
    print(f"从 {knowledge_file} 读取知识库...")
    knowledge_base = read_knowledge_base(knowledge_file)
    print(f"共读取到 {len(knowledge_base)} 个知识块")
    retriever = TaskOnlyKnowledgeRetriever(knowledge_base)
    
    # 外层循环：重复运行两次
    for run_number in range(2, 4):  # 运行两次，run_number从1到2
        print(f"\n=== 第{run_number}次运行开始 ===")
        
        # 内层循环：遍历三个模型
        for model_name in models:
            print(f"\n--- 使用模型 {model_name} 进行分析 ---")
            
            # 处理案例
            print("开始分析案例...")
            results = process_cases(cases, retriever, model_name, api_delay)
            
            # 生成输出文件名，包含模型名称和运行次数
            model_short_name = model_name.replace("-", "").replace("_", "")
            output_file = f"{model_short_name}二分类任务_无RAG_第{run_number}次运行.xlsx"
            
            # 保存结果
            save_results(results, output_file)
            
            print(f"模型 {model_name} 第{run_number}次运行完成")
        
        print(f"=== 第{run_number}次运行完成 ===\n")
    
    print("所有模型的所有运行已完成！")

if __name__ == "__main__":
    main()