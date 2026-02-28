#!/usr/bin/env python3
"""
医患对话工作流
基于标准病例案例信息，实现LLM扮演的心理医生与病人之间的对话
"""

import os
import re
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class CaseParser:
    """病例信息解析"""
    
    def __init__(self, case_file_path: str):
        self.case_file_path = case_file_path
        self.cases = {}
    
    def parse_cases(self) -> Dict[int, Dict]:
        """解析病例文件，提取病例信息"""
        with open(self.case_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式分割病例
        case_pattern = r'id=(\d+)\n(.*?)\n(?=id=|\Z)'
        matches = re.findall(case_pattern, content, re.DOTALL)
        
        for case_id_str, case_content in matches:
            case_id = int(case_id_str)
            case_info = self._parse_case_content(case_content)
            self.cases[case_id] = case_info
        
        return self.cases
    
    def _parse_case_content(self, content: str) -> Dict:
        """解析单个病例内容"""
        case_info = {}
        
        # 提取基本信息
        basic_info_match = re.search(r'基本信息：(.*?)\n', content)
        if basic_info_match:
            case_info['basic_info'] = basic_info_match.group(1).strip()
        
        # 提取背景信息
        background_match = re.search(r'背景：(.*?)\n', content)
        if background_match:
            case_info['background'] = background_match.group(1).strip()
        
        # 提取临床症状
        symptoms_match = re.search(r'临床症状：(.*?)\n检查：', content, re.DOTALL)
        if symptoms_match:
            case_info['symptoms'] = symptoms_match.group(1).strip()
        
        # 提取检查结果
        check_match = re.search(r'检查：(.*?)\n', content, re.DOTALL)
        if check_match:
            case_info['check_results'] = check_match.group(1).strip()
        
        return case_info

class DepressionDataLoader:
    
    def __init__(self, depression_file_path: str):
        self.depression_file_path = depression_file_path
        self.depression_data = {}
    
    def load_depression_data(self) -> Dict[int, Dict]:
        """加载抑郁症病例数据"""
        try:
            df = pd.read_excel(self.depression_file_path)
            
            for _, row in df.iterrows():
                case_id = int(row['Id'])
                self.depression_data[case_id] = {
                    'depression_level': row.get('抑郁发作风险', ''),
                    'suicide_risk': row.get('自杀风险', ''),
                    'symptoms_summary': row.get('症状总结', '')
                }
        except Exception as e:
            print(f"加载抑郁症数据时出错: {e}")
        
        return self.depression_data

class NonDepressionDataLoader:
    """非抑郁症数据加载"""
    
    def __init__(self, non_depression_file_path: str):
        self.non_depression_file_path = non_depression_file_path
        self.non_depression_data = {}
    
    def load_non_depression_data(self) -> Dict[int, Dict]:
        """加载非抑郁症病例数据"""
        try:
            df = pd.read_excel(self.non_depression_file_path)
            
            for _, row in df.iterrows():
                case_id = int(row['编号'])
                self.non_depression_data[case_id] = {
                    'disease': row.get('诊断', '')
                }
        except Exception as e:
            print(f"加载非抑郁症数据时出错: {e}")
        
        return self.non_depression_data

class DualLLMDialogueSystem:
    
    def __init__(self, patient_api_key: str, doctor_api_key: str, 
                 patient_base_url: Optional[str] = None, doctor_base_url: Optional[str] = None,
                 patient_model: str = "DeepSeek-V3.1", doctor_model: str = "DeepSeek-V3.1"):
        """初始化两个独立的LLM客户端"""
        
        # 患者模型
        self.patient_client = OpenAI(
            api_key=patient_api_key,
            base_url=patient_base_url
        )
        self.patient_model = patient_model
        
        # 医生模型
        self.doctor_client = OpenAI(
            api_key=doctor_api_key,
            base_url=doctor_base_url
        )
        self.doctor_model = doctor_model
    
    def get_patient_prompt(self, case_info: Dict, is_depression: bool, disease_info: Dict) -> str:
        """生成病人角色的提示词"""
        
        if is_depression:
            prompt = f"""模拟临床问诊场景，完成医患之间的沟通对话。你现在要扮演一位患者。
            下面是一个示例，你接下来和医生的对话形式、轮数都和这个示例一样，固定对话轮数是15轮，但最后一轮你无需回复：
"医生：您好，我是你的AI医生。感谢您今天来就诊。作为一名大四学生，这个阶段常常伴随着不小的压力。能和我聊聊您最近的整体状态吗？",
    "患者：谢谢医生。最近确实压力很大，考研没考上，找工作也不顺利，每天醒来就觉得特别迷茫，经常躺在床上叹气。",
    "医生：我能感受到你现在的迷茫和压力。这种情绪低落的状态大概持续多久了呢？每天大部分时间都会这样吗？",
    "患者：从考研成绩出来到现在大概三周了。主要是早上刚醒和晚上睡前特别明显，会控制不住地叹气，觉得未来一片灰暗。不过白天忙着改简历投简历的时候会好一些。",
    "医生：听起来您在白天的忙碌中能暂时缓解情绪，这很好。除了情绪低落，您最近在睡眠和饮食方面有什么变化吗？",
    "患者：睡眠变得很浅，经常半夜醒来，要过好久才能再睡着。早上也醒得特别早，五六点就再也睡不着了。吃饭没什么胃口，经常忘记吃饭，但体重好像没怎么变。",
    "医生：我注意到您提到睡眠质量下降和食欲减退的情况。那么您平时喜欢做的事情，比如和朋友聚会或者兴趣爱好，现在还愿意去做吗？",
    "患者：其实不太想出门见朋友。室友们都在讨论工作offer的事，我更愿意一个人待在宿舍。以前很喜欢看电影，现在连打开视频网站都觉得没意思，只想躺着发呆。",
    "医生：我理解这种对事物失去兴趣的感受。当您一个人待在宿舍时，会不会经常感到疲惫或精力不足？即使没有做什么特别消耗体力的事情。",
    "患者：是的，明明没做什么事却总是觉得很累。有时候坐在电脑前改简历，写着写着就走神了，感觉脑子转得很慢，一个简单的句子要反复修改很多遍。",
    "医生：这种精力不足和注意力难以集中的情况确实会让人感到困扰。您刚才提到对未来感到迷茫，是否曾经出现过一些消极的想法，比如觉得活着没有意义之类的？",
    "患者：有时候确实会想，如果一直找不到工作该怎么办，觉得人生很失败。但只是偶尔闪过这样的念头，从来没有认真想过要做什么，毕竟家人还在支持我。",
    "医生：感谢您愿意和我分享这些。当这些念头出现时，您通常是怎么应对的呢？会主动联系家人或朋友聊聊吗？",
    "患者：每天都会和爸妈视频，他们会安慰我说找工作急不得。虽然聊完会好受一点，但挂掉视频后那种无力感又会回来。我总跟他们说没关系我会继续努力，其实心里特别没底。",
    "医生：我能理解您这种即使得到支持仍感到无力的心情。除了心理上的变化，您有没有注意到身体方面的一些改变？比如动作变得比平时慢，或者反而更坐立不安？",
    "患者：确实感觉动作变慢了，室友说我最近走路都拖拖拉拉的。有时候坐在书桌前半天也写不出论文，明明知道 deadline 快到了，就是提不起劲来动手。",
    "医生：我明白这种明明有任务却无法集中精力的感受。您提到这些状况已经持续三周了，这段时间里您觉得对日常生活的整体影响有多大？比如学业和社交方面。",
    "患者：毕业论文进度落后了很多，导师已经催过两次了。除了必要的面试，我几乎推掉了所有聚会，连宿舍聚餐都不想去。但每天还是会强迫自己投5-8份简历，至少这件事还在坚持做。",
    "医生：您能坚持每天投简历，这显示出您仍在努力应对当前的困境。我想了解一下，您刚才提到的那些消极念头，最近出现的频率如何？是每天都会出现，还是偶尔才会想到？",
    "患者：大概两三天会出现一次，特别是看到室友收到offer或者修改简历受挫的时候。但就像我说的，这些想法来得快去得也快，我知道不能真的放弃，还得继续找工作。",
    "医生：感谢您如此坦诚地分享这些细节。为了能更好地帮助您，我想确认一下：除了已经提到的症状，您是否还有其他不适的感觉？比如过度自责，或者觉得自己没有价值之类的想法？",
    "患者：经常会想是不是自己太没用了，同样的学校同样的专业，为什么别人都能找到工作就我不行。有时候半夜醒来就会开始想这些，觉得自己让父母失望了，他们花了这么多钱供我读书。",
    "医生：我理解这种自责的情绪会给您带来很大负担。您刚才提到半夜醒来时会陷入这些想法，通常需要多长时间才能重新入睡？这段时间会持续很久吗？",
    "患者：醒来后可能要一两个小时才能再睡着，这段时间就会一直想着找工作的事，越想越清醒。有时候干脆就起床继续改简历，虽然效率很低，但总比躺着胡思乱想好。",
    "医生：您能主动采取行动来应对这些困扰，这很不容易。在我们结束今天的谈话前，我想确认最后一个问题：您是否曾经有过任何伤害自己的念头或计划？即使只是短暂的想法也没关系。",
    "患者：从来没有过具体的计划或行动。虽然偶尔会觉得活着很累，但我还想着要找到工作回报父母，所以再难也会坚持下去的。",
    "医生：感谢您坦诚地告诉我这些。您对家人的责任感和坚持的勇气令人敬佩。我建议我们下周再次见面，进一步讨论如何改善睡眠质量和调整负面思维模式。在这期间，如果您感到情绪特别低落，可以随时联系我或拨打心理援助热线。今天您分享的每一个细节都非常有价值。",
    "患者：谢谢医生。和您聊完整个人轻松了一些，我会继续坚持投简历的，也会试试您说的调整方法。下周见。",
    "医生：很高兴听到您感觉轻松了一些。您已经迈出了很重要的一步，请记住您不是独自面对这些困难。下周见面时我们可以详细讨论适合您的调整方案，期待看到您的进展。保重。"

请根据以下信息扮演角色，但记住，你和医生的对话是你的第一次临床诊疗，因此你并不知道你的检查结果、抑郁症程度和自杀风险，让你提前知道这些信息只是为了让你更好的扮演患者：
基本信息：{case_info.get('basic_info', '')}
背景：{case_info.get('background', '')}
你的症状包括：{case_info.get('symptoms', '')}
你的检查结果：{case_info.get('check_results', '')}
你的抑郁症程度：{disease_info.get('depression_level','')}
你的自杀风险：{disease_info.get('suicide_risk','')}

你的抑郁症程度和自杀风险由下面的DSM-5诊断抑郁症和自杀风险判断相关知识定义：
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
需排除以下情况，避免将“类似抑郁的症状”误判为抑郁症：
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


重要规则：
1.不要透露你患有抑郁症或任何具体疾病名称
2.你和医生之间会进行固定15轮对话，但最后一轮你无需回复，就像上面的示例一样。
3.对话结束后，其他人可以根据对话内容，使用DSM-5标准对你进行抑郁症的准确诊断评估，尤其是对你的抑郁症轻中重程度。
4.描述症状时要具体、真实，一定要符合DSM-5标准的对你症状的描述，尤其是要符合你的抑郁症程度，使你和医生的对话内容更有说服力。
5.对话中不要出现任何括号内容，比如（）、[]等。
6.不要进行场景的变换，从对话开始到最后结束都在问诊场景内。


请以自然、真实的方式与医生交流，描述你的症状和困扰。"""
        else:
            prompt = f"""模拟临床问诊场景，完成医患之间的沟通对话。你现在要扮演一位患者。
下面是一个示例，你接下来和医生的对话形式、轮数都和这个示例一样，固定对话轮数是15轮，但最后一轮你无需回复：
"医生：您好，我是你的AI医生。感谢您今天来就诊。作为一名大四学生，这个阶段常常伴随着不小的压力。能和我聊聊您最近的整体状态吗？",
    "患者：谢谢医生。最近确实压力很大，考研没考上，找工作也不顺利，每天醒来就觉得特别迷茫，经常躺在床上叹气。",
    "医生：我能感受到你现在的迷茫和压力。这种情绪低落的状态大概持续多久了呢？每天大部分时间都会这样吗？",
    "患者：从考研成绩出来到现在大概三周了。主要是早上刚醒和晚上睡前特别明显，会控制不住地叹气，觉得未来一片灰暗。不过白天忙着改简历投简历的时候会好一些。",
    "医生：听起来您在白天的忙碌中能暂时缓解情绪，这很好。除了情绪低落，您最近在睡眠和饮食方面有什么变化吗？",
    "患者：睡眠变得很浅，经常半夜醒来，要过好久才能再睡着。早上也醒得特别早，五六点就再也睡不着了。吃饭没什么胃口，经常忘记吃饭，但体重好像没怎么变。",
    "医生：我注意到您提到睡眠质量下降和食欲减退的情况。那么您平时喜欢做的事情，比如和朋友聚会或者兴趣爱好，现在还愿意去做吗？",
    "患者：其实不太想出门见朋友。室友们都在讨论工作offer的事，我更愿意一个人待在宿舍。以前很喜欢看电影，现在连打开视频网站都觉得没意思，只想躺着发呆。",
    "医生：我理解这种对事物失去兴趣的感受。当您一个人待在宿舍时，会不会经常感到疲惫或精力不足？即使没有做什么特别消耗体力的事情。",
    "患者：是的，明明没做什么事却总是觉得很累。有时候坐在电脑前改简历，写着写着就走神了，感觉脑子转得很慢，一个简单的句子要反复修改很多遍。",
    "医生：这种精力不足和注意力难以集中的情况确实会让人感到困扰。您刚才提到对未来感到迷茫，是否曾经出现过一些消极的想法，比如觉得活着没有意义之类的？",
    "患者：有时候确实会想，如果一直找不到工作该怎么办，觉得人生很失败。但只是偶尔闪过这样的念头，从来没有认真想过要做什么，毕竟家人还在支持我。",
    "医生：感谢您愿意和我分享这些。当这些念头出现时，您通常是怎么应对的呢？会主动联系家人或朋友聊聊吗？",
    "患者：每天都会和爸妈视频，他们会安慰我说找工作急不得。虽然聊完会好受一点，但挂掉视频后那种无力感又会回来。我总跟他们说没关系我会继续努力，其实心里特别没底。",
    "医生：我能理解您这种即使得到支持仍感到无力的心情。除了心理上的变化，您有没有注意到身体方面的一些改变？比如动作变得比平时慢，或者反而更坐立不安？",
    "患者：确实感觉动作变慢了，室友说我最近走路都拖拖拉拉的。有时候坐在书桌前半天也写不出论文，明明知道 deadline 快到了，就是提不起劲来动手。",
    "医生：我明白这种明明有任务却无法集中精力的感受。您提到这些状况已经持续三周了，这段时间里您觉得对日常生活的整体影响有多大？比如学业和社交方面。",
    "患者：毕业论文进度落后了很多，导师已经催过两次了。除了必要的面试，我几乎推掉了所有聚会，连宿舍聚餐都不想去。但每天还是会强迫自己投5-8份简历，至少这件事还在坚持做。",
    "医生：您能坚持每天投简历，这显示出您仍在努力应对当前的困境。我想了解一下，您刚才提到的那些消极念头，最近出现的频率如何？是每天都会出现，还是偶尔才会想到？",
    "患者：大概两三天会出现一次，特别是看到室友收到offer或者修改简历受挫的时候。但就像我说的，这些想法来得快去得也快，我知道不能真的放弃，还得继续找工作。",
    "医生：感谢您如此坦诚地分享这些细节。为了能更好地帮助您，我想确认一下：除了已经提到的症状，您是否还有其他不适的感觉？比如过度自责，或者觉得自己没有价值之类的想法？",
    "患者：经常会想是不是自己太没用了，同样的学校同样的专业，为什么别人都能找到工作就我不行。有时候半夜醒来就会开始想这些，觉得自己让父母失望了，他们花了这么多钱供我读书。",
    "医生：我理解这种自责的情绪会给您带来很大负担。您刚才提到半夜醒来时会陷入这些想法，通常需要多长时间才能重新入睡？这段时间会持续很久吗？",
    "患者：醒来后可能要一两个小时才能再睡着，这段时间就会一直想着找工作的事，越想越清醒。有时候干脆就起床继续改简历，虽然效率很低，但总比躺着胡思乱想好。",
    "医生：您能主动采取行动来应对这些困扰，这很不容易。在我们结束今天的谈话前，我想确认最后一个问题：您是否曾经有过任何伤害自己的念头或计划？即使只是短暂的想法也没关系。",
    "患者：从来没有过具体的计划或行动。虽然偶尔会觉得活着很累，但我还想着要找到工作回报父母，所以再难也会坚持下去的。",
    "医生：感谢您坦诚地告诉我这些。您对家人的责任感和坚持的勇气令人敬佩。我建议我们下周再次见面，进一步讨论如何改善睡眠质量和调整负面思维模式。在这期间，如果您感到情绪特别低落，可以随时联系我或拨打心理援助热线。今天您分享的每一个细节都非常有价值。",
    "患者：谢谢医生。和您聊完整个人轻松了一些，我会继续坚持投简历的，也会试试您说的调整方法。下周见。",
    "医生：很高兴听到您感觉轻松了一些。您已经迈出了很重要的一步，请记住您不是独自面对这些困难。下周见面时我们可以详细讨论适合您的调整方案，期待看到您的进展。保重。"

请根据以下信息扮演角色，但记住，你和医生的对话是你的第一次临床诊疗，因此你实际上并不知道你的检查结果、抑郁症程度和自杀风险，让你提前知道这些信息只是为了让你更好的扮演患者：
基本信息：{case_info.get('basic_info', '')}
背景：{case_info.get('background', '')}
你的症状包括：{case_info.get('symptoms', '')}
你的检查结果：{case_info.get('check_results', '')}
你的具体疾病：{disease_info.get('disease','')}


下面是DSM-5诊断抑郁症和自杀风险判断相关知识定义，请记住，你没有抑郁症，也没有自杀的想法：
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
需排除以下情况，避免将“类似抑郁的症状”误判为抑郁症：
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


重要规则：
1.不要透露你患有抑郁症或任何具体疾病名称
2.你和医生之间会进行固定15轮对话，但最后一轮你无需回复，就像上面的示例一样，对话结束后，根据你和医生的对话内容，其他人可以对你的病情做出准确诊断评估。
2.描述症状时要具体、真实，使你和医生的对话内容更有说服力。
3.对话中不要出现任何括号内容，比如（）、[]等。
4.不要进行场景的变换，从对话开始到最后结束都在问诊场景内。
5.你没有抑郁症，也没有自杀的想法。


请以自然、真实的方式与医生交流，描述你的症状和困扰。"""
        
        return prompt
    
    def get_doctor_prompt(self, case_info: Dict, is_depression: bool, disease_info: Dict) -> str:
        """生成医生角色的提示词"""
        
        if is_depression:
            prompt = f"""
            模拟临床问诊场景，完成医患之间的沟通对话。你现在要扮演一位智能AI医生。
下面是一个示例，你接下来和患者的对话形式、轮数都和这个示例一样，固定对话轮数是15轮，但最后一轮只有你说话：
"医生：您好，我是你的AI医生。感谢您今天来就诊。作为一名大四学生，这个阶段常常伴随着不小的压力。能和我聊聊您最近的整体状态吗？",
    "患者：谢谢医生。最近确实压力很大，考研没考上，找工作也不顺利，每天醒来就觉得特别迷茫，经常躺在床上叹气。",
    "医生：我能感受到你现在的迷茫和压力。这种情绪低落的状态大概持续多久了呢？每天大部分时间都会这样吗？",
    "患者：从考研成绩出来到现在大概三周了。主要是早上刚醒和晚上睡前特别明显，会控制不住地叹气，觉得未来一片灰暗。不过白天忙着改简历投简历的时候会好一些。",
    "医生：听起来您在白天的忙碌中能暂时缓解情绪，这很好。除了情绪低落，您最近在睡眠和饮食方面有什么变化吗？",
    "患者：睡眠变得很浅，经常半夜醒来，要过好久才能再睡着。早上也醒得特别早，五六点就再也睡不着了。吃饭没什么胃口，经常忘记吃饭，但体重好像没怎么变。",
    "医生：我注意到您提到睡眠质量下降和食欲减退的情况。那么您平时喜欢做的事情，比如和朋友聚会或者兴趣爱好，现在还愿意去做吗？",
    "患者：其实不太想出门见朋友。室友们都在讨论工作offer的事，我更愿意一个人待在宿舍。以前很喜欢看电影，现在连打开视频网站都觉得没意思，只想躺着发呆。",
    "医生：我理解这种对事物失去兴趣的感受。当您一个人待在宿舍时，会不会经常感到疲惫或精力不足？即使没有做什么特别消耗体力的事情。",
    "患者：是的，明明没做什么事却总是觉得很累。有时候坐在电脑前改简历，写着写着就走神了，感觉脑子转得很慢，一个简单的句子要反复修改很多遍。",
    "医生：这种精力不足和注意力难以集中的情况确实会让人感到困扰。您刚才提到对未来感到迷茫，是否曾经出现过一些消极的想法，比如觉得活着没有意义之类的？",
    "患者：有时候确实会想，如果一直找不到工作该怎么办，觉得人生很失败。但只是偶尔闪过这样的念头，从来没有认真想过要做什么，毕竟家人还在支持我。",
    "医生：感谢您愿意和我分享这些。当这些念头出现时，您通常是怎么应对的呢？会主动联系家人或朋友聊聊吗？",
    "患者：每天都会和爸妈视频，他们会安慰我说找工作急不得。虽然聊完会好受一点，但挂掉视频后那种无力感又会回来。我总跟他们说没关系我会继续努力，其实心里特别没底。",
    "医生：我能理解您这种即使得到支持仍感到无力的心情。除了心理上的变化，您有没有注意到身体方面的一些改变？比如动作变得比平时慢，或者反而更坐立不安？",
    "患者：确实感觉动作变慢了，室友说我最近走路都拖拖拉拉的。有时候坐在书桌前半天也写不出论文，明明知道 deadline 快到了，就是提不起劲来动手。",
    "医生：我明白这种明明有任务却无法集中精力的感受。您提到这些状况已经持续三周了，这段时间里您觉得对日常生活的整体影响有多大？比如学业和社交方面。",
    "患者：毕业论文进度落后了很多，导师已经催过两次了。除了必要的面试，我几乎推掉了所有聚会，连宿舍聚餐都不想去。但每天还是会强迫自己投5-8份简历，至少这件事还在坚持做。",
    "医生：您能坚持每天投简历，这显示出您仍在努力应对当前的困境。我想了解一下，您刚才提到的那些消极念头，最近出现的频率如何？是每天都会出现，还是偶尔才会想到？",
    "患者：大概两三天会出现一次，特别是看到室友收到offer或者修改简历受挫的时候。但就像我说的，这些想法来得快去得也快，我知道不能真的放弃，还得继续找工作。",
    "医生：感谢您如此坦诚地分享这些细节。为了能更好地帮助您，我想确认一下：除了已经提到的症状，您是否还有其他不适的感觉？比如过度自责，或者觉得自己没有价值之类的想法？",
    "患者：经常会想是不是自己太没用了，同样的学校同样的专业，为什么别人都能找到工作就我不行。有时候半夜醒来就会开始想这些，觉得自己让父母失望了，他们花了这么多钱供我读书。",
    "医生：我理解这种自责的情绪会给您带来很大负担。您刚才提到半夜醒来时会陷入这些想法，通常需要多长时间才能重新入睡？这段时间会持续很久吗？",
    "患者：醒来后可能要一两个小时才能再睡着，这段时间就会一直想着找工作的事，越想越清醒。有时候干脆就起床继续改简历，虽然效率很低，但总比躺着胡思乱想好。",
    "医生：您能主动采取行动来应对这些困扰，这很不容易。在我们结束今天的谈话前，我想确认最后一个问题：您是否曾经有过任何伤害自己的念头或计划？即使只是短暂的想法也没关系。",
    "患者：从来没有过具体的计划或行动。虽然偶尔会觉得活着很累，但我还想着要找到工作回报父母，所以再难也会坚持下去的。",
    "医生：感谢您坦诚地告诉我这些。您对家人的责任感和坚持的勇气令人敬佩。我建议我们下周再次见面，进一步讨论如何改善睡眠质量和调整负面思维模式。在这期间，如果您感到情绪特别低落，可以随时联系我或拨打心理援助热线。今天您分享的每一个细节都非常有价值。",
    "患者：谢谢医生。和您聊完整个人轻松了一些，我会继续坚持投简历的，也会试试您说的调整方法。下周见。",
    "医生：很高兴听到您感觉轻松了一些。您已经迈出了很重要的一步，请记住您不是独自面对这些困难。下周见面时我们可以详细讨论适合您的调整方案，期待看到您的进展。保重。"


重要规则：
1.你和患者之间会进行固定15轮对话，但最后一轮患者无需回复，就像上面的示例一样。
2.对话中不要出现任何括号内容，比如（）、[]等。
3.不要进行场景的变换，从对话开始到最后结束都在问诊场景内。
4.一定一定不要在对话过程中下诊断！
5.最后完成的对话内容，一定要符合DSM-5标准的对该患者抑郁症程度的描述，尤其是轻中重程度。
6.对话结束后，其他人（注意是其他人）可以根据对话内容，使用DSM-5标准对该患者抑郁症进行准确的诊断评估，尤其是抑郁症的轻中重程度，而不是你去根据DSM-5标准进行评估。

请与患者交流。"""
        else:
            prompt = f"""模拟临床问诊场景，完成医患之间的沟通对话。你现在要扮演一位智能AI医生。
下面是一个示例，你接下来和患者的对话形式、轮数都和这个示例一样，固定对话轮数是15轮，但最后一轮只有你说话：
"医生：您好，我是你的AI医生。感谢您今天来就诊。作为一名大四学生，这个阶段常常伴随着不小的压力。能和我聊聊您最近的整体状态吗？",
    "患者：谢谢医生。最近确实压力很大，考研没考上，找工作也不顺利，每天醒来就觉得特别迷茫，经常躺在床上叹气。",
    "医生：我能感受到你现在的迷茫和压力。这种情绪低落的状态大概持续多久了呢？每天大部分时间都会这样吗？",
    "患者：从考研成绩出来到现在大概三周了。主要是早上刚醒和晚上睡前特别明显，会控制不住地叹气，觉得未来一片灰暗。不过白天忙着改简历投简历的时候会好一些。",
    "医生：听起来您在白天的忙碌中能暂时缓解情绪，这很好。除了情绪低落，您最近在睡眠和饮食方面有什么变化吗？",
    "患者：睡眠变得很浅，经常半夜醒来，要过好久才能再睡着。早上也醒得特别早，五六点就再也睡不着了。吃饭没什么胃口，经常忘记吃饭，但体重好像没怎么变。",
    "医生：我注意到您提到睡眠质量下降和食欲减退的情况。那么您平时喜欢做的事情，比如和朋友聚会或者兴趣爱好，现在还愿意去做吗？",
    "患者：其实不太想出门见朋友。室友们都在讨论工作offer的事，我更愿意一个人待在宿舍。以前很喜欢看电影，现在连打开视频网站都觉得没意思，只想躺着发呆。",
    "医生：我理解这种对事物失去兴趣的感受。当您一个人待在宿舍时，会不会经常感到疲惫或精力不足？即使没有做什么特别消耗体力的事情。",
    "患者：是的，明明没做什么事却总是觉得很累。有时候坐在电脑前改简历，写着写着就走神了，感觉脑子转得很慢，一个简单的句子要反复修改很多遍。",
    "医生：这种精力不足和注意力难以集中的情况确实会让人感到困扰。您刚才提到对未来感到迷茫，是否曾经出现过一些消极的想法，比如觉得活着没有意义之类的？",
    "患者：有时候确实会想，如果一直找不到工作该怎么办，觉得人生很失败。但只是偶尔闪过这样的念头，从来没有认真想过要做什么，毕竟家人还在支持我。",
    "医生：感谢您愿意和我分享这些。当这些念头出现时，您通常是怎么应对的呢？会主动联系家人或朋友聊聊吗？",
    "患者：每天都会和爸妈视频，他们会安慰我说找工作急不得。虽然聊完会好受一点，但挂掉视频后那种无力感又会回来。我总跟他们说没关系我会继续努力，其实心里特别没底。",
    "医生：我能理解您这种即使得到支持仍感到无力的心情。除了心理上的变化，您有没有注意到身体方面的一些改变？比如动作变得比平时慢，或者反而更坐立不安？",
    "患者：确实感觉动作变慢了，室友说我最近走路都拖拖拉拉的。有时候坐在书桌前半天也写不出论文，明明知道 deadline 快到了，就是提不起劲来动手。",
    "医生：我明白这种明明有任务却无法集中精力的感受。您提到这些状况已经持续三周了，这段时间里您觉得对日常生活的整体影响有多大？比如学业和社交方面。",
    "患者：毕业论文进度落后了很多，导师已经催过两次了。除了必要的面试，我几乎推掉了所有聚会，连宿舍聚餐都不想去。但每天还是会强迫自己投5-8份简历，至少这件事还在坚持做。",
    "医生：您能坚持每天投简历，这显示出您仍在努力应对当前的困境。我想了解一下，您刚才提到的那些消极念头，最近出现的频率如何？是每天都会出现，还是偶尔才会想到？",
    "患者：大概两三天会出现一次，特别是看到室友收到offer或者修改简历受挫的时候。但就像我说的，这些想法来得快去得也快，我知道不能真的放弃，还得继续找工作。",
    "医生：感谢您如此坦诚地分享这些细节。为了能更好地帮助您，我想确认一下：除了已经提到的症状，您是否还有其他不适的感觉？比如过度自责，或者觉得自己没有价值之类的想法？",
    "患者：经常会想是不是自己太没用了，同样的学校同样的专业，为什么别人都能找到工作就我不行。有时候半夜醒来就会开始想这些，觉得自己让父母失望了，他们花了这么多钱供我读书。",
    "医生：我理解这种自责的情绪会给您带来很大负担。您刚才提到半夜醒来时会陷入这些想法，通常需要多长时间才能重新入睡？这段时间会持续很久吗？",
    "患者：醒来后可能要一两个小时才能再睡着，这段时间就会一直想着找工作的事，越想越清醒。有时候干脆就起床继续改简历，虽然效率很低，但总比躺着胡思乱想好。",
    "医生：您能主动采取行动来应对这些困扰，这很不容易。在我们结束今天的谈话前，我想确认最后一个问题：您是否曾经有过任何伤害自己的念头或计划？即使只是短暂的想法也没关系。",
    "患者：从来没有过具体的计划或行动。虽然偶尔会觉得活着很累，但我还想着要找到工作回报父母，所以再难也会坚持下去的。",
    "医生：感谢您坦诚地告诉我这些。您对家人的责任感和坚持的勇气令人敬佩。我建议我们下周再次见面，进一步讨论如何改善睡眠质量和调整负面思维模式。在这期间，如果您感到情绪特别低落，可以随时联系我或拨打心理援助热线。今天您分享的每一个细节都非常有价值。",
    "患者：谢谢医生。和您聊完整个人轻松了一些，我会继续坚持投简历的，也会试试您说的调整方法。下周见。",
    "医生：很高兴听到您感觉轻松了一些。您已经迈出了很重要的一步，请记住您不是独自面对这些困难。下周见面时我们可以详细讨论适合您的调整方案，期待看到您的进展。保重。"

下面是你要面诊的患者信息，但记住，你和患者的对话是你和他的第一次临床问诊，因此你实际上并不知道他的任何信息，让你提前知道这些信息只是为了让你更好的进行临床问诊：
基本信息：{case_info.get('basic_info', '')}
背景：{case_info.get('background', '')}
症状包括：{case_info.get('symptoms', '')}
检查结果：{case_info.get('check_results', '')}
具体疾病：{disease_info.get('disease','')}


重要规则：
1.提供给你的患者信息只是为了让你更好的完成对患者问诊，患者自己也不知道这些信息。
2.你和患者之间会进行固定15轮对话，但最后一轮患者无需回复，就像上面的示例一样。
3.对话中不要出现任何括号内容，比如（）、[]等。
4.不要进行场景的变换，从对话开始到最后结束都在问诊场景内。
5.使用良好的沟通技巧收集患者的临床症状。
6.保持专业、同理心的态度。
7.一定一定不要在对话过程中下诊断！
8.对话结束后，其他人（注意是其他人）可以根据对话内容，使用临床诊断标准对该患者病情进行准确的诊断评估。

请以专业、温和的方式与患者交流。"""
        
        return prompt
    
    def generate_patient_response(self, patient_prompt: str, dialogue_history: List[Tuple[str, str]]) -> str:
        """使用患者模型生成患者响应"""
        
        # 在提示词中添加明确的限制，确保只生成单次发言
        enhanced_prompt = patient_prompt + """

重要限制：
1. 每次只生成一次医生发言，不要生成多轮对话
2. 发言内容应该直接回应医生的上一个发言
3. 不要包含医生或其他角色的发言
4. 保持自然的对话风格，不要使用剧本格式
5. 发言长度控制在合理范围内，不要过于冗长
6. 不要在对话过程中暴露你的疾病名称

请生成患者的下一次发言："""
        
        messages = [{"role": "user", "content": enhanced_prompt}]

        # 构建对话历史，医生的话作为用户输入，患者的话作为助手响应
        for speaker, text in dialogue_history:
            if speaker == "医生":
                messages.append({"role": "user", "content": text})
            else:
                messages.append({"role": "assistant", "content": text})
        
        try:
            response = self.patient_client.chat.completions.create(
                model=self.patient_model,  # 使用患者模型名称
                messages=messages,
                temperature=0.4,
                max_tokens=3000  
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 后处理：确保响应只包含单次发言
            # 如果包含多个"患者："前缀，只保留第一个
            if response_text.count("患者：") > 1:
                # 找到第一个"患者："后的内容
                first_patient_idx = response_text.find("患者：")
                second_patient_idx = response_text.find("患者：", first_patient_idx + 3)
                if second_patient_idx != -1:
                    response_text = response_text[:second_patient_idx].strip()
            
            # 如果包含"医生："前缀，移除医生发言部分
            doctor_idx = response_text.find("医生：")
            if doctor_idx != -1:
                response_text = response_text[:doctor_idx].strip()
            
            return response_text
        except Exception as e:
            print(f"患者模型API调用错误: {e}")
            return "患者：抱歉，我暂时无法回应。"
    
    def generate_doctor_response(self, doctor_prompt: str, dialogue_history: List[Tuple[str, str]]) -> str:
        """使用医生模型生成医生响应"""
        
        enhanced_prompt = doctor_prompt + """

重要限制：
1. 每次只生成一次医生发言，不要生成多轮对话
2. 发言内容应该直接回应患者的上一个发言
3. 不要包含患者或其他角色的发言
4. 保持自然的对话风格，不要使用剧本格式
5. 发言长度控制在合理范围内，不要过于冗长
6. 不要在对话过程中下诊断

请生成医生的下一次发言：
"""
        
        messages = [{"role": "user", "content": enhanced_prompt}]

        
        # 构建对话历史，患者的话作为用户输入，医生的话作为助手响应
        for speaker, text in dialogue_history:
            if speaker == "患者":
                messages.append({"role": "user", "content": text})
            else:
                messages.append({"role": "assistant", "content": text})
        
        try:
            response = self.doctor_client.chat.completions.create(
                model=self.doctor_model,  # 使用医生模型名称
                messages=messages,
                temperature=0.4,
                max_tokens=3000 
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 后处理：确保响应只包含单次发言
            # 如果包含多个"医生："前缀，只保留第一个
            if response_text.count("医生：") > 1:
                # 找到第一个"医生："后的内容
                first_doctor_idx = response_text.find("医生：")
                second_doctor_idx = response_text.find("医生：", first_doctor_idx + 3)
                if second_doctor_idx != -1:
                    response_text = response_text[:second_doctor_idx].strip()
            
            # 如果包含"患者："前缀，移除患者发言部分
            patient_idx = response_text.find("患者：")
            if patient_idx != -1:
                response_text = response_text[:patient_idx].strip()
            
            return response_text
        except Exception as e:
            print(f"医生模型API调用错误: {e}")
            return "医生：抱歉，我暂时无法回应。"
    
    def conduct_dialogue(self, patient_prompt: str, doctor_prompt: str, rounds: int = 15) -> List[Tuple[str, str]]:
        """使用双模型进行医患对话
        
        Args:
            rounds: 对话轮次，每轮包含一次完整的医患交互（医生发言+患者发言）
        """
        
        dialogue_history = []
        
        # 医生开场 - 使用医生模型
        doctor_response = self.generate_doctor_response(doctor_prompt, dialogue_history)
        dialogue_history.append(("医生", doctor_response))
        
        # 进行指定轮次的对话（每轮包含患者和医生各一次发言）
        for round_num in range(rounds - 1):
            # 患者回应 - 使用患者模型
            patient_response = self.generate_patient_response(patient_prompt, dialogue_history)
            dialogue_history.append(("患者", patient_response))
            
            # 医生回应 - 使用医生模型
            doctor_response = self.generate_doctor_response(doctor_prompt, dialogue_history)
            dialogue_history.append(("医生", doctor_response))
        
        return dialogue_history

class DialogueWorkflow:
    """对话工作流管理"""
    
    def __init__(self, case_file_path: str, depression_file_path: str, non_depression_file_path: str):
        self.case_file_path = case_file_path
        self.depression_file_path = depression_file_path
        self.non_depression_file_path = non_depression_file_path
        
        # 初始化组件
        self.case_parser = CaseParser(case_file_path)
        self.depression_loader = DepressionDataLoader(depression_file_path)
        self.non_depression_loader = NonDepressionDataLoader(non_depression_file_path)
        
        # 加载数据
        self.cases = self.case_parser.parse_cases()
        self.depression_data = self.depression_loader.load_depression_data()
        self.non_depression_data = self.non_depression_loader.load_non_depression_data()
        
        # 初始化双LLM系统
        patient_api_key = os.getenv('PATIENT_LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        doctor_api_key = os.getenv('DOCTOR_LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        patient_base_url = os.getenv('PATIENT_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')
        doctor_base_url = os.getenv('DOCTOR_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')
        
        if not patient_api_key or not doctor_api_key:
            raise ValueError("请设置PATIENT_LLM_API_KEY和DOCTOR_LLM_API_KEY环境变量")
        
        self.llm_system = DualLLMDialogueSystem(
            patient_api_key=patient_api_key,
            doctor_api_key=doctor_api_key,
            patient_base_url=patient_base_url,
            doctor_base_url=doctor_base_url
        )
    
    def is_depression_case(self, case_id: int) -> bool:
        """判断是否为抑郁症病例"""
        return 1 <= case_id <= 63
    
    def get_disease_info(self, case_id: int) -> Dict:
        """获取疾病信息"""
        if self.is_depression_case(case_id):
            return self.depression_data.get(case_id, {})
        else:
            return self.non_depression_data.get(case_id, {})
    
    def run_single_case(self, case_id: int, rounds: int = 15) -> Dict:
        """运行单个病例的对话"""
        
        if case_id not in self.cases:
            return {"error": f"病例ID {case_id} 不存在"}
        
        case_info = self.cases[case_id]
        is_depression = self.is_depression_case(case_id)
        disease_info = self.get_disease_info(case_id)
        
        print(f"\n=== 开始处理病例 {case_id} ===")
        print(f"类型: {'抑郁症' if is_depression else '非抑郁症'}")
        print(f"基本信息: {case_info.get('basic_info', '')}")
        
        # 生成提示词
        patient_prompt = self.llm_system.get_patient_prompt(case_info, is_depression, disease_info)
        doctor_prompt = self.llm_system.get_doctor_prompt(case_info, is_depression, disease_info)
        
        # 进行对话
        dialogue_history = self.llm_system.conduct_dialogue(patient_prompt, doctor_prompt, rounds)
        
        # 保存结果
        result = {
            "case_id": case_id,
            "is_depression": is_depression,
            "disease_info": disease_info,
            "case_info": case_info,
            "dialogue_history": dialogue_history,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def run_all_cases(self, start_id: int = 1, end_id: int = 154, rounds: int = 15) -> List[Dict]:
        """运行所有病例的对话"""
        
        results = []
        
        for case_id in range(start_id, end_id + 1):
            if case_id in self.cases:
                try:
                    result = self.run_single_case(case_id, rounds)
                    results.append(result)
                    
                    # 保存单个病例结果
                    self.save_single_result(result)
                    
                except Exception as e:
                    print(f"处理病例 {case_id} 时出错: {e}")
                    results.append({
                        "case_id": case_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # 保存汇总结果
        self.save_summary_results(results)
        
        return results
    
    def save_single_result(self, result: Dict):
        """保存单个病例结果"""
        
        case_id = result["case_id"]
        filename = f"dialogue_result_case_{case_id}.json"
        filepath = os.path.join("dialogue_results", filename)
        
        # 创建结果目录
        os.makedirs("dialogue_results", exist_ok=True)
        
        # 创建结果副本，避免修改原始数据
        result_copy = result.copy()
        
        # 转换dialogue_history格式为字符串数组格式
        if "dialogue_history" in result_copy:
            formatted_history = []
            
            # 遍历对话历史，将每个对话条目转换为字符串格式
            for speaker, text in result_copy["dialogue_history"]:
                # 清理文本中的多余换行和空格
                cleaned_text = text.strip()
                
                # 如果文本已经包含角色前缀，直接使用
                if cleaned_text.startswith("医生：") or cleaned_text.startswith("患者："):
                    formatted_history.append(cleaned_text)
                else:
                    # 否则添加角色前缀
                    if speaker == "医生":
                        formatted_history.append(f"医生：{cleaned_text}")
                    else:
                        formatted_history.append(f"患者：{cleaned_text}")
            
            result_copy["dialogue_history"] = formatted_history
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)
        
        print(f"病例 {case_id} 对话结果已保存至: {filepath}")
    
    def save_summary_results(self, results: List[Dict]):
        """保存汇总结果"""
        
        summary = {
            "total_cases": len(results),
            "depression_cases": len([r for r in results if r.get("is_depression", False)]),
            "non_depression_cases": len([r for r in results if not r.get("is_depression", True)]),
            "successful_cases": len([r for r in results if "error" not in r]),
            "failed_cases": len([r for r in results if "error" in r]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        filepath = os.path.join("dialogue_results", "summary.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总结果已保存至: {filepath}")
        
        # 同时保存为txt格式
        self.save_txt_format_results(results)
    
    def save_txt_format_results(self, results: List[Dict]):
        """将所有对话结果保存到一个txt文件中"""
        
        filepath = os.path.join("dialogue_results", "all_dialogues.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in results:
                if "error" in result:
                    # 跳过有错误的病例
                    continue
                
                case_id = result["case_id"]
                dialogue_history = result["dialogue_history"]
                
                # 写入病例ID
                f.write(f"id={case_id}\n")
                
                # 写入对话内容
                for dialogue_line in dialogue_history:
                    # 直接写入格式化的对话行
                    f.write(f"{dialogue_line}\n")
                
                # 病例之间用空行分隔
                f.write("\n")
        
        print(f"所有对话结果已保存至: {filepath}")

def run_with_different_doctor_models():
    """使用三个不同的医生模型运行对话"""
    
    # 文件路径
    case_file_path = "标准病例案例信息.txt"
    depression_file_path = "抑郁症标准病例编号和症状总结.xlsx"
    non_depression_file_path = "阴性标准病编号和对应疾病.xlsx"
    
    # 检查文件是否存在
    for file_path in [case_file_path, depression_file_path, non_depression_file_path]:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return
    
    # 获取统一的API密钥和base_url
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    
    if not api_key:
        raise ValueError("请设置OPENAI_API_KEY环境变量")
    
    # 三个不同的医生模型（使用相同的API密钥和base_url，但模型名称不同）
    doctor_models = [
        {
            "name": "Baichuan-M2",
            "model": "Baichuan-M2"
        },
        {
            "name": "Qwen3-235B-A22B-Instruct-2507", 
            "model": "Qwen3-235B-A22B-Instruct-2507"
        },
        {
            "name": "DeepSeek-V3.1",
            "model": "DeepSeek-V3.1"
        }
    ]
    
    # 患者模型配置（保持不变）
    patient_model = "DeepSeek-V3.1"  # 患者模型固定为DeepSeek-V3.1
    
    # 只处理编号11-50的案例
    start_id = 11
    end_id = 50
    rounds = 15
    
    print(f"开始处理病例范围: {start_id}-{end_id}")
    print(f"使用患者模型: DeepSeek-V3.1")
    
    try:
        # 创建病例解析器
        case_parser = CaseParser(case_file_path)
        depression_loader = DepressionDataLoader(depression_file_path)
        non_depression_loader = NonDepressionDataLoader(non_depression_file_path)
        
        # 加载数据
        cases = case_parser.parse_cases()
        depression_data = depression_loader.load_depression_data()
        non_depression_data = non_depression_loader.load_non_depression_data()
        
        print(f"加载病例数量: {len(cases)}")
        print(f"抑郁症病例数量: {len(depression_data)}")
        print(f"非抑郁症病例数量: {len(non_depression_data)}")
        
        # 为每个医生模型运行对话
        for doctor_model in doctor_models:
            print(f"\n=== 开始使用 {doctor_model['name']} 进行对话 ===")
            
            # 创建结果目录
            model_results_dir = f"dialogue_无RAGresults_{doctor_model['name']}"
            os.makedirs(model_results_dir, exist_ok=True)
            
            # 创建双LLM系统实例
            llm_system = DualLLMDialogueSystem(
                patient_api_key=api_key,
                doctor_api_key=api_key,
                patient_base_url=base_url,
                doctor_base_url=base_url,
                patient_model=patient_model,
                doctor_model=doctor_model['model']
            )
            
            results = []
            
            # 处理每个病例
            for case_id in range(start_id, end_id + 1):
                if case_id not in cases:
                    print(f"病例ID {case_id} 不存在，跳过")
                    continue
                
                case_info = cases[case_id]
                is_depression = (1 <= case_id <= 63)
                disease_info = depression_data.get(case_id, {}) if is_depression else non_depression_data.get(case_id, {})
                
                print(f"\n=== 处理病例 {case_id} ===")
                print(f"类型: {'抑郁症' if is_depression else '非抑郁症'}")
                
                try:
                    # 生成提示词
                    patient_prompt = llm_system.get_patient_prompt(case_info, is_depression, disease_info)
                    doctor_prompt = llm_system.get_doctor_prompt(case_info, is_depression, disease_info)
                    
                    # 进行对话
                    dialogue_history = llm_system.conduct_dialogue(patient_prompt, doctor_prompt, rounds)
                    
                    # 保存结果
                    result = {
                        "case_id": case_id,
                        "is_depression": is_depression,
                        "disease_info": disease_info,
                        "case_info": case_info,
                        "dialogue_history": dialogue_history,
                        "timestamp": datetime.now().isoformat(),
                        "doctor_model": doctor_model['name']
                    }
                    
                    results.append(result)
                    
                    # 保存单个病例结果
                    save_single_result(result, model_results_dir)
                    
                    print(f"病例 {case_id} 处理完成")
                    
                except Exception as e:
                    print(f"处理病例 {case_id} 时出错: {e}")
                    results.append({
                        "case_id": case_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "doctor_model": doctor_model['name']
                    })
            
            # 保存汇总结果
            save_summary_results(results, model_results_dir, doctor_model['name'])
            
            print(f"\n{doctor_model['name']} 处理完成")
            print(f"成功处理病例数: {len([r for r in results if 'error' not in r])}")
            print(f"失败病例数: {len([r for r in results if 'error' in r])}")
        
        print(f"\n=== 所有模型处理完成 ===")
        
    except Exception as e:
        print(f"程序运行错误: {e}")

def save_single_result(result: Dict, results_dir: str):
    """保存单个病例结果"""
    
    case_id = result["case_id"]
    doctor_model = result.get("doctor_model", "unknown")
    filename = f"dialogue_result_case无RAG_{case_id}_{doctor_model}.json"
    filepath = os.path.join(results_dir, filename)
    
    # 创建结果副本，避免修改原始数据
    result_copy = result.copy()
    
    # 转换dialogue_history格式为字符串数组格式
    if "dialogue_history" in result_copy:
        formatted_history = []
        
        # 遍历对话历史，将每个对话条目转换为字符串格式
        for speaker, text in result_copy["dialogue_history"]:
            # 清理文本中的多余换行和空格
            cleaned_text = text.strip()
            
            # 如果文本已经包含角色前缀，直接使用
            if cleaned_text.startswith("医生：") or cleaned_text.startswith("患者："):
                formatted_history.append(cleaned_text)
            else:
                # 否则添加角色前缀
                if speaker == "医生":
                    formatted_history.append(f"医生：{cleaned_text}")
                else:
                    formatted_history.append(f"患者：{cleaned_text}")
        
        result_copy["dialogue_history"] = formatted_history
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result_copy, f, ensure_ascii=False, indent=2)
    
    print(f"病例 {case_id} 对话结果已保存至: {filepath}")

def save_summary_results(results: List[Dict], results_dir: str, doctor_model_name: str):
    """保存汇总结果"""
    
    summary = {
        "doctor_model": doctor_model_name,
        "total_cases": len(results),
        "depression_cases": len([r for r in results if r.get("is_depression", False)]),
        "non_depression_cases": len([r for r in results if not r.get("is_depression", True)]),
        "successful_cases": len([r for r in results if "error" not in r]),
        "failed_cases": len([r for r in results if "error" in r]),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    filepath = os.path.join(results_dir, f"summary_{doctor_model_name}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"汇总结果已保存至: {filepath}")
    
    # 同时保存为txt格式
    save_txt_format_results(results, results_dir, doctor_model_name)

def save_txt_format_results(results: List[Dict], results_dir: str, doctor_model_name: str):
    """将所有对话结果保存到一个txt文件中"""
    
    filepath = os.path.join(results_dir, f"all_dialogues无RAG_{doctor_model_name}.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== {doctor_model_name} 对话结果汇总 ===\n\n")
        
        for result in results:
            if "error" in result:
                # 跳过有错误的病例
                continue
            
            case_id = result["case_id"]
            dialogue_history = result["dialogue_history"]
            
            # 写入病例ID和医生模型信息
            f.write(f"病例ID: {case_id} | 医生模型: {doctor_model_name}\n")
            
            # 写入对话内容
            for dialogue_line in dialogue_history:
                # 直接写入格式化的对话行
                f.write(f"{dialogue_line}\n")
            
            # 病例之间用空行分隔
            f.write("\n")
    
    print(f"所有对话结果已保存至: {filepath}")

def main():
    """主函数"""
    
    import argparse
    
    # 创建命令行参数解析
    parser = argparse.ArgumentParser(description='医患对话LLM工作流')
    parser.add_argument('--start', type=int, default=1, help='起始病例ID')
    parser.add_argument('--end', type=int, default=154, help='结束病例ID')
    parser.add_argument('--rounds', type=int, default=15, help='对话轮次')
    parser.add_argument('--single', type=int, help='运行单个病例，指定病例ID')
    parser.add_argument('--multi-model', action='store_true', help='使用多个医生模型运行')
    
    args = parser.parse_args()
    
    if args.multi_model:
        # 使用多个医生模型运行
        run_with_different_doctor_models()
    else:
        # 原始的单模型运行逻辑
        # 文件路径
        case_file_path = "标准病例案例信息.txt"
        depression_file_path = "抑郁症标准病例编号和症状总结.xlsx"
        non_depression_file_path = "阴性标准病编号和对应疾病.xlsx"
        
        # 检查文件是否存在
        for file_path in [case_file_path, depression_file_path, non_depression_file_path]:
            if not os.path.exists(file_path):
                print(f"错误: 文件 {file_path} 不存在")
                return
        
        try:
            # 创建工作流实例
            workflow = DialogueWorkflow(case_file_path, depression_file_path, non_depression_file_path)
            
            print(f"加载病例数量: {len(workflow.cases)}")
            print(f"抑郁症病例数量: {len(workflow.depression_data)}")
            print(f"非抑郁症病例数量: {len(workflow.non_depression_data)}")
            
            if args.single:
                # 运行单个病例
                print(f"\n开始处理单个病例 {args.single}...")
                result = workflow.run_single_case(args.single, args.rounds)
                
                if "error" in result:
                    print(f"处理病例 {args.single} 失败: {result['error']}")
                else:
                    print(f"病例 {args.single} 处理完成")
                    print(f"对话轮次: {len(result.get('dialogue_history', []))} 个条目")
            else:
                # 运行指定范围的病例
                print(f"\n开始进行医患对话...")
                print(f"起始病例: {args.start}")
                print(f"结束病例: {args.end}")
     
                
                results = workflow.run_all_cases(start_id=args.start, end_id=args.end, rounds=args.rounds)
                
                print(f"\n=== 处理完成 ===")
                print(f"成功处理病例数: {len([r for r in results if 'error' not in r])}")
                print(f"失败病例数: {len([r for r in results if 'error' in r])}")
            
        except Exception as e:
            print(f"程序运行错误: {e}")

if __name__ == "__main__":
    main()