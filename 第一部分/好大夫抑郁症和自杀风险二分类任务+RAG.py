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
    你是一位心理医生，擅长判断抑郁症和评估自杀风险。
    请分析以下医患沟通对话，完成两项判断：
    1. 判断患者是否有抑郁症。请仅回答"有"或"无"
    2. 患者是否有自杀风险？请仅回答"有"或"无"

患者抑郁症程度是由DSM-5诊断标准确定的，下面是DSM-5诊断标准和自杀评估的相关知识：
主题：DSM-5诊断标准鉴别患者是否存在抑郁症
摘选：### 一、如何鉴别患者是否存在抑郁症（重性抑郁障碍）
DSM-5对抑郁症的鉴别核心是“是否符合重性抑郁发作的诊断标准”，需通过三步判断，同时满足“症状要求”“功能损害”和“排除干扰”三大条件：
#### 第一步：判断是否符合“重性抑郁发作”的症状与持续时间
重性抑郁发作需同时满足三个维度：
1. **时间维度**：症状需在同一2周时期内持续存在，且几乎每天大部分时间都能观察到或患者能主观感受到。
2. **症状数量与核心症状**：需出现以下9项症状中的**至少5项**，且这5项中必须包含**至少1项核心症状**（抑郁症的核心特征）：
   - 核心症状（2选1，必含1项）：
     - 持续的情绪低落（主观描述“悲伤、空虚、绝望”，或他人观察到“表情呆滞、频繁哭泣”）；
     - 对几乎所有活动的兴趣或愉悦感显著下降（如“以前喜欢的爱好现在完全提不起劲”“做什么都觉得没意思”）。
   - 其他伴随症状（任选，与核心症状共凑5项）：
     - 体重或食欲显著变化（2周内体重变化≥5%，或持续“没胃口吃不下”“暴饮暴食控制不住”）；
     - 睡眠障碍（几乎每天失眠，或每天嗜睡，且严重影响白天状态）；
     - 精神运动性异常（他人可观察到的激越：坐立不安、小动作增多；或迟滞：语速变慢、动作迟缓、反应迟钝）；
     - 疲劳或精力不足（即使没做体力劳动，也常感到“浑身乏力、没精神”，且休息后无法缓解）；
     - 无价值感或过度内疚（如反复自责“都是我的错”，对小事过度愧疚，甚至出现“自己不配活着”的妄想性内疚）；
     - 思维能力下降（注意力难以集中、记忆力减退，做简单决定都很困难，如“选早餐要纠结半小时”）；
     - 反复出现死亡或自杀相关想法（从“活着没意义”的模糊念头，到有明确计划的自杀意念，甚至自杀尝试）。
#### 第二步：判断症状是否导致“临床显著损害”
上述症状必须对患者的**社交、职业或其他重要功能领域**造成明显困扰或损害，而非轻微影响。例如：
- 职业功能：无法正常上班，频繁请假，工作效率大幅下降，甚至因状态差被辞退；
- 社交功能：回避所有社交活动，与亲友关系疏远，频繁发生争吵；
- 日常生活：无法自理基本生活，如不做饭、不打扫卫生、不规律洗漱。
若症状仅轻微干扰生活，未达到“显著损害”程度，则不满足诊断条件。
#### 第三步：排除其他干扰因素（排除标准）
需排除以下情况，避免将“类似depression的症状”误判为抑郁症：
1. **排除物质或躯体疾病诱因**：症状由酒精、药物（如镇静剂、降压药）滥用/戒断，或甲状腺功能减退、帕金森病、脑瘤、慢性疼痛等躯体疾病直接导致（需通过医学检查明确排除）；
2. **排除其他精神障碍的“抑郁表现”**：例如，双相障碍患者的“抑郁发作”需先排除既往是否有“躁狂/轻躁狂发作”；精神分裂症患者的抑郁症状可能是精神病性症状的伴随表现，需优先诊断精神病性障碍；
3. **排除正常的“丧亲反应”**：亲人去世后的悲伤情绪通常在1-2个月内逐渐缓解，且不会出现“无价值感”“过度内疚”“与思念无关的自杀意念”等症状；若悲伤持续超2周且符合抑郁症核心症状，才需考虑诊断。

主题：DSM-5抑郁症轻度、中度、重度鉴别诊断标准
摘选：### 二、抑郁症（重性抑郁发作）的轻、中、重度分类
DSM-5根据“**症状数量**”“**功能损害程度**”和“**是否存在精神病性特征**”，将重性抑郁发作分为三个等级，核心区别在于症状对个体生活的影响范围和严重程度：
#### 1. 轻度重性抑郁发作
- **核心判断标准**：
  1. 症状数量：刚达到诊断标准（仅5项症状，且包含1项核心症状）；
  2. 功能损害：社交、职业等功能仅轻微受损，基本活动仍能维持；
  3. 无精神病性特征（无幻觉、妄想等精神病性症状）。
- **具体表现示例**：每天情绪低落，对爱好失去兴趣，睡眠变差（每天少睡1-2小时），工作时注意力易分散，但能完成基本工作任务；周末不愿参加聚会，但会回复亲友消息，日常生活能自理。
#### 2. 中度重性抑郁发作
- **核心判断标准**：
  1. 症状数量：症状数量介于轻度与重度之间（通常6-7项，包含1项核心症状）；
  2. 功能损害：社交、职业等功能显著受损，基本活动需强迫自己才能维持；
  3. 无精神病性特征。
- **具体表现示例**：情绪持续低落，对所有活动都提不起劲，2周内体重下降6%，每天仅睡3-4小时，浑身乏力，做“吃什么、穿什么”等简单决定都要纠结1小时以上；上班时频繁走神，多次被领导批评，只能勉强完成部分工作；拒绝所有社交邀请，与家人交流极少，需家人提醒才会洗漱、吃饭。
#### 3. 重度重性抑郁发作
- **核心判断标准**：
  1. 症状数量：症状数量接近或达到9项（通常同时包含2项核心症状，伴随症状全面出现）；
  2. 功能损害：社交、职业等功能严重受损，甚至无法维持基本日常生活；
  3. 可分为“无精神病性特征”和“伴精神病性特征”两类（后者更严重）。
- **具体表现示例**：
  - 无精神病性特征：情绪极度低落，常独自哭泣或表情呆滞，对一切事物都无兴趣，2周内体重下降10%，每天仅睡2小时或嗜睡超12小时，动作迟缓到“走路需他人搀扶”，反复出现“活着没用”的想法，完全无法上班，卧床时间超过半天，无法自理做饭、洗澡等基本生活；
  - 伴精神病性特征：在上述表现基础上，出现与抑郁主题相关的幻觉（如听到“你该死，别拖累家人”的声音）或妄想（如坚信“自己是家庭的累赘，家人早就想害死我”），且这些精神病性症状严重影响认知和行为。

主题：自杀风险判别标准
摘选：自杀风险量化：
-0：无自杀意念
-1：被动自杀意念无计划
-2：有具体计划未实施
-3：近期自杀未遂

    
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
    models = ["DeepSeek-V3.2-Exp"]
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
    for run_number in range(1, 2):  # 运行两次，run_number从1到2
        print(f"\n=== 第{run_number}次运行开始 ===")
        
        # 内层循环：遍历三个模型
        for model_name in models:
            print(f"\n--- 使用模型 {model_name} 进行分析 ---")
            
            # 处理案例
            print("开始分析案例...")
            results = process_cases(cases, retriever, model_name, api_delay)
            
            # 生成输出文件名，包含模型名称和运行次数
            model_short_name = model_name.replace("-", "").replace("_", "")
            output_file = f"{model_short_name}二分类任务_有RAG_第{run_number}次运行.xlsx"
            
            # 保存结果
            save_results(results, output_file)
            
            print(f"模型 {model_name} 第{run_number}次运行完成")
        
        print(f"=== 第{run_number}次运行完成 ===\n")
    
    print("所有模型的所有运行已完成！")

if __name__ == "__main__":
    main()