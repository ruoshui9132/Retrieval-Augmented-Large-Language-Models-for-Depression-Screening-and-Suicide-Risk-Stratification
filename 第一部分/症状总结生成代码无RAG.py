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

# 构建提示词 - 调整为生成症状总结的要求
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
     下面是一个症状总结示例：
     来访者（37岁女性）近半年情绪低落（罩玻璃罐），兴趣丧失（拒拆化妆品），精力减退（醒后更累）。伴自我评价低（扇耳光）、睡眠障碍（多梦易醒）、自杀意念（"活着没意思"）。工作请假增多。PHQ-14分。建议立即精神科急诊就诊，严格防范自杀风险，启动危机干预。

    仿照这个示例，回答中请只生成症状总结内容，不要添加任何其他内容，这一点非常重要。

    
    医患沟通对话内容：
    {case_content}
    """
    return prompt.strip()

# 调用LLM API生成症状总结
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
                max_tokens=10000     # 限制输出长度
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

# 批量处理案例
def process_cases(cases: Dict[int, str], retriever: TaskOnlyKnowledgeRetriever, model_name: str = "Qwen3-235B-A22B-Instruct-2507", delay_seconds: float = 1.0) -> Dict[int, str]:
    """批量处理所有案例，返回包含案例ID和对应总结的字典"""
    results = {}
    total = len(cases)
    
    for i, (case_id, content) in enumerate(sorted(cases.items(), key=lambda x: x[0])):
        print(f"使用模型 {model_name} 处理案例 {case_id} ({i+1}/{total})...")
        
        summary = generate_summary(content, retriever, model_name)
        results[case_id] = summary
        
        # 添加延迟，避免触发API速率限制
        if i < total - 1:
            time.sleep(delay_seconds)
    
    return results

# 保存结果到文本文件 - 完全重写以符合指定格式
def save_results(results: Dict[int, str], output_file: str):
    """将症状总结按指定格式保存到文本文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # 按案例ID排序
        for case_id in sorted(results.keys()):
            f.write(f"id={case_id}\n")
            f.write(f"{results[case_id]}\n\n")  # 每个案例后空一行分隔
    
 
# 主函数
def main():
    # 配置参数
    input_file = "好大夫200例抑郁症对话.txt"       # 包含案例的文本文件路径
    knowledge_file = "知识库.txt"             # 知识库文件路径
    api_delay = 1.0                           # API调用之间的延迟（秒）
    
    # 定义要使用的模型列表
    models = ["DeepSeek-R1"]
    
    # 外层循环：运行2次
    for run_number in range(1, 2):
        print(f"\n=== 开始第{run_number}次运行 ===")
        
        # 内层循环：依次使用三个模型
        for model_name in models:
            print(f"\n--- 使用模型 {model_name} 进行处理 ---")
            
            # 读取案例
            print(f"从 {input_file} 读取案例...")
            cases = read_cases(input_file)
            print(f"共读取到 {len(cases)} 个案例")
            
            # 读取知识库并创建检索器
            print(f"从 {knowledge_file} 读取知识库...")
            knowledge_base = read_knowledge_base(knowledge_file)
            print(f"共读取到 {len(knowledge_base)} 个知识块")
            retriever = TaskOnlyKnowledgeRetriever(knowledge_base)
            
            # 处理案例
            print(f"开始使用模型 {model_name} 生成症状总结...")
            results = process_cases(cases, retriever, model_name, api_delay)
            
            # 生成输出文件名，包含模型名称和运行次数
            model_short_name = model_name.replace("-", "").replace(" ", "")[:10]  # 简化模型名称
            output_file = f"症状总结结果无RAG_{model_short_name}_第{run_number}次运行.txt"
            
            # 保存结果
            save_results(results, output_file)
            
            print(f"模型 {model_name} 第{run_number}次运行完成")
    
    print("\n=== 所有运行完成 ===")

if __name__ == "__main__":
    main()
