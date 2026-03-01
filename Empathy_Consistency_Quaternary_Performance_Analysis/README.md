# Empathy Consistency and Quaternary Classification Performance Analysis

This repository contains analysis scripts and data for two key evaluation components of LLM-based mental health assessment:

1. **Empathy Score Inter-rater Reliability Analysis** - Evaluating consistency of human raters' assessments
2. **Quaternary Classification Performance Analysis** - Class-specific performance metrics for depression and suicide risk stratification

## Project Structure

```
Empathy_Consistency_Quaternary_Performance_Analysis/
├── scripts/                          # Analysis scripts
│   ├── empathy_consistency_analysis.py    # Inter-rater reliability analysis
│   └── classification_analysis.py         # Classification task analysis
├── data/                             # Data files
│   ├── empathy_evaluation/           # Empathy evaluation data
│   └── classification_tasks/         # Classification task data
│       ├── haodf_binary/             # Haodf binary classification
│       ├── haodf_quaternary/         # Haodf 4-class classification
│       ├── standard_binary/          # Standard case binary classification
│       └── standard_quaternary/      # Standard case 4-class classification
├── results/                          # Analysis results
│   ├── empathy_consistency/          # Empathy consistency results
│   └── classification_analysis/      # Classification analysis results
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Analysis Components

### 1. Empathy Score Inter-rater Reliability Analysis

Evaluates the consistency of human raters' assessments of LLM empathy across three dimensions:
- **Emotional Matching**: How well the response matches the patient's emotional state
- **Dialogue Flow Consistency**: Natural progression of conversation
- **Appropriate Care Expression**: Suitable demonstration of care and concern

**Key Metrics:**
- ICC(2,1): Intraclass Correlation Coefficient for ordinal data
- Fleiss' Kappa: Multi-rater agreement for categorical data
- Kendall's W: Coefficient of concordance

**Main Finding:** Inter-rater reliability reached **Excellent** level (mean ICC = 0.849)

### 2. Quaternary Classification Performance Analysis

Analyzes class-specific performance for fine-grained stratification:

| Task | Classes | Description |
|------|---------|-------------|
| Depression 4-Class | 4 | None / Mild / Moderate / Severe |
| Suicide Risk 4-Class | 4 | No Risk / Low Risk / Medium Risk / High Risk |

**Datasets:**
- **Haodf Dataset**: Real-world clinical dialogues from Haodf online healthcare platform
- **Standard Case Dataset**: Standardized clinical case vignettes

**Models Evaluated:**
- Baichuan-M2
- DeepSeek-R1
- Qwen3-235B-A22B-Instruct-2507

**Conditions:**
- With RAG (Retrieval-Augmented Generation)
- Without RAG

## Quick Start

### Environment Setup

```bash
pip install -r requirements.txt
```

### Run Analysis

```bash
# Run empathy consistency analysis
python scripts/empathy_consistency_analysis.py

# Run classification analysis
python scripts/classification_analysis.py
```

## Key Results Summary

### Empathy Evaluation Reliability

| Metric | Mean | Interpretation |
|--------|------|----------------|
| ICC(2,1) | 0.849 | Excellent |
| Kendall's W | 0.423 | Moderate agreement |

### Quaternary Classification Performance

#### Depression 4-Class (Best Results)

| Dataset | Model | Condition | Accuracy | Macro F1 |
|---------|-------|-----------|----------|----------|
| Haodf | Qwen3-235B | With RAG | 0.893 | 0.854 |
| Standard | Qwen3-235B | With RAG | 0.855 | 0.773 |

#### Suicide Risk 4-Class (Best Results)

| Dataset | Model | Condition | Accuracy | Macro F1 |
|---------|-------|-----------|----------|----------|
| Haodf | Qwen3-235B | With RAG | 0.882 | 0.743 |
| Standard | Qwen3-235B | With RAG | 0.955 | 0.924 |

### Class-Specific Performance Insights

The quaternary classification analysis reveals:
- **None/No Risk class**: Highest precision and recall across models
- **Moderate/Medium Risk class**: Generally good recall but lower precision
- **Mild/Low Risk class**: Most challenging to classify accurately
- **Severe/High Risk class**: High precision but variable recall

## Data Format

### Empathy Evaluation Data
- Excel files with rater scores (1-5 Likert scale)
- Each row represents one dialogue evaluation
- Columns include rater IDs and scores for each metric

### Classification Task Data
- Excel files with model predictions
- Standard labels and model predictions in separate columns
- Multiple runs (3 runs) for reliability

## Citation

If you use this code or data, please cite the associated paper.

## License

This project is for academic research purposes only.
