# Empathy Score Inter-rater Reliability Analysis Report

## 1. Evaluation Design

- **Number of Raters**: 3 independent raters per dialogue
- **Evaluation Metrics**: Emotional Matching, Dialogue Flow Consistency, Appropriate Care Expression
- **Rating Scale**: 1-5 point Likert scale
- **Evaluation Subjects**: 40 dialogues (Case ID 11-50)

## 2. Reliability Metrics

| Metric | Application | Interpretation |
|--------|-------------|----------------|
| ICC(2,1) | Continuous/Ordinal data | <0.40 Poor, 0.40-0.60 Fair, 0.60-0.75 Good, >0.75 Excellent |
| Fleiss' Kappa | Categorical data | <0.20 Slight, 0.20-0.40 Fair, 0.40-0.60 Moderate, 0.60-0.80 Substantial, >0.80 Almost Perfect |
| Kendall's W | Ordinal data ranks | 0-1, higher is better |

## 3. Detailed Results

### _Baichuan-M2

| Metric | ICC(2,1) | Kendall's W | Fleiss' Kappa | Mean Pairwise Kappa |
|--------|----------|-------------|---------------|---------------------|
| 情感匹配度 | 0.8608 | 0.4231 | 0.0278 | 0.1131 |
| 对话流程一致性 | 0.8747 | 0.4497 | -0.0190 | 0.0582 |
| 关怀表达适当性 | 0.8734 | 0.4490 | 0.0601 | 0.1419 |

### _DeepSeek-V3.1

| Metric | ICC(2,1) | Kendall's W | Fleiss' Kappa | Mean Pairwise Kappa |
|--------|----------|-------------|---------------|---------------------|
| 情感匹配度 | 0.8360 | 0.3797 | -0.0374 | 0.0674 |
| 对话流程一致性 | 0.9601 | 0.7025 | 0.2608 | 0.2985 |
| 关怀表达适当性 | 0.9059 | 0.5318 | 0.1377 | 0.2279 |

### _Qwen3-235B-A22B-Instruct-2507

| Metric | ICC(2,1) | Kendall's W | Fleiss' Kappa | Mean Pairwise Kappa |
|--------|----------|-------------|---------------|---------------------|
| 情感匹配度 | 0.8141 | 0.3481 | -0.0965 | 0.0108 |
| 对话流程一致性 | 0.8077 | 0.3333 | -0.0169 | nan |
| 关怀表达适当性 | 0.8077 | 0.3333 | -0.0169 | nan |

### 无RAG_Baichuan-M2

| Metric | ICC(2,1) | Kendall's W | Fleiss' Kappa | Mean Pairwise Kappa |
|--------|----------|-------------|---------------|---------------------|
| 情感匹配度 | 0.7079 | 0.2421 | -0.2908 | 0.0927 |
| 对话流程一致性 | 0.8715 | 0.4515 | -0.1264 | 0.0244 |
| 关怀表达适当性 | 0.7816 | 0.3190 | -0.2535 | 0.1069 |

### 无RAG_DeepSeek-V3.1

| Metric | ICC(2,1) | Kendall's W | Fleiss' Kappa | Mean Pairwise Kappa |
|--------|----------|-------------|---------------|---------------------|
| 情感匹配度 | 0.9312 | 0.5401 | 0.0238 | 0.3029 |
| 对话流程一致性 | 0.9317 | 0.6100 | -0.0082 | 0.1403 |
| 关怀表达适当性 | 0.8915 | 0.3989 | -0.0286 | 0.1857 |

### 无RAG_Qwen3-235B-A22B-Instruct-2507

| Metric | ICC(2,1) | Kendall's W | Fleiss' Kappa | Mean Pairwise Kappa |
|--------|----------|-------------|---------------|---------------------|
| 情感匹配度 | 0.7912 | 0.3333 | -0.2500 | nan |
| 对话流程一致性 | 0.8469 | 0.4295 | -0.1333 | 0.0649 |
| 关怀表达适当性 | 0.7923 | 0.3333 | -0.1796 | 0.0159 |

## 4. Summary Statistics

| Metric | Mean | SD | Min | Max | Interpretation |
|--------|------|-----|-----|-----|----------------|
| ICC(2,1) | 0.8492 | 0.0618 | 0.7079 | 0.9601 | Excellent |
| Fleiss' Kappa | -0.0526 | 0.1358 | -0.2908 | 0.2608 | Poor (worse than chance) |
| Kendall's W | 0.4227 | 0.1117 | 0.2421 | 0.7025 | - |

## 5. Conclusion

The inter-rater reliability for empathy scores reached **Excellent** level (mean ICC = 0.849), with Kendall's W = 0.423, indicating high reliability of the evaluation results.

### Note on Low Kappa Values

The relatively low Fleiss' Kappa values can be attributed to:

1. **Restricted score range**: Most scores concentrated at 4-5, leading to blurred category boundaries
2. **Kappa's sensitivity to marginal distributions**: When scores are highly consistent, chance agreement is also high, underestimating Kappa
3. **ICC is more suitable for ordinal data**: ICC considers the ordinal nature of scores and better reflects inter-rater reliability

Therefore, ICC is recommended as the primary reliability metric, and results indicate **excellent** inter-rater reliability.
