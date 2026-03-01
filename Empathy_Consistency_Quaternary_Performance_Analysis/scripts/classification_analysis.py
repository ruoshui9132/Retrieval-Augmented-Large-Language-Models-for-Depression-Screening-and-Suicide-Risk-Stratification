# -*- coding: utf-8 -*-
"""
Classification Task Analysis Script
Includes: Depression binary, Depression 4-class, Suicide risk binary, Suicide risk 4-class
Merges 3 runs to eliminate random effects

Author: Research Team
Purpose: Analyze classification performance for depression and suicide risk assessment
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Use sans-serif font family for better compatibility
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Class labels (English)
BINARY_LABELS = ['No', 'Yes']
DEPRESSION_QUAD_LABELS = ['None(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)']
DEPRESSION_QUAD_NAMES = ['None', 'Mild', 'Moderate', 'Severe']
SUICIDE_QUAD_LABELS = ['No Risk(0)', 'Low Risk(1)', 'Medium Risk(2)', 'High Risk(3)']
SUICIDE_QUAD_NAMES = ['No Risk', 'Low Risk', 'Medium Risk', 'High Risk']

# Depression class mapping
DEPRESSION_MAP = {'无': 0, '轻度': 1, '中度': 2, '重度': 3}

# RAG condition mapping (Chinese to English)
RAG_MAP = {'无RAG': 'Without RAG', '有RAG': 'With RAG'}


def load_binary_data(base_path, folder_name):
    """
    Load binary classification task data
    
    Returns:
        dict: Contains depression and suicide risk data
    """
    folder_path = os.path.join(base_path, folder_name)
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    
    data_dict = {}
    
    for file in files:
        parts = file.replace('.xlsx', '').split('_')
        model = parts[0].replace('二分类任务', '').replace('标准病例', '')
        rag_condition = RAG_MAP.get(parts[1], parts[1])
        run_num = parts[2]
        
        key = f"{model}_{rag_condition}"
        if key not in data_dict:
            data_dict[key] = {
                'depression': [],
                'suicide_risk': []
            }
        
        df = pd.read_excel(os.path.join(folder_path, file))
        
        # Depression binary classification
        y_true_dep = (df['有无抑郁症'] == '有').astype(int).values
        y_pred_dep = (df['抑郁症'] == '有').astype(int).values
        
        # Suicide risk binary classification (0=no risk, 1/2/3=has risk)
        y_true_suicide = (df['有无自杀风险'] == '有').astype(int).values
        y_pred_suicide_raw = df['自杀风险'].values
        y_pred_suicide = (y_pred_suicide_raw >= 1).astype(int)
        
        data_dict[key]['depression'].append({
            'run': run_num,
            'y_true': y_true_dep,
            'y_pred': y_pred_dep
        })
        
        data_dict[key]['suicide_risk'].append({
            'run': run_num,
            'y_true': y_true_suicide,
            'y_pred': y_pred_suicide
        })
    
    return data_dict


def load_quad_data(base_path, folder_name):
    """
    Load 4-class classification task data (depression and suicide risk)
    
    Returns:
        dict: Contains depression and suicide risk 4-class data
    """
    folder_path = os.path.join(base_path, folder_name)
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    
    data_dict = {}
    
    for file in files:
        parts = file.replace('.xlsx', '').split('_')
        model = parts[0].replace('四分类任务', '').replace('标准病例', '')
        rag_condition = RAG_MAP.get(parts[1], parts[1])
        run_num = parts[2]
        
        key = f"{model}_{rag_condition}"
        if key not in data_dict:
            data_dict[key] = {
                'depression': [],
                'suicide_risk': []
            }
        
        df = pd.read_excel(os.path.join(folder_path, file))
        
        # Depression 4-class: convert text labels to numbers
        y_true_dep = df['抑郁症（标准）'].map(DEPRESSION_MAP).values
        y_pred_dep = df['抑郁症'].map(DEPRESSION_MAP).values
        
        # Suicide risk 4-class: already numeric
        y_true_suicide = df['自杀风险（标准）'].values
        y_pred_suicide = df['自杀风险'].values
        
        # Handle potential NaN values
        valid_mask_dep = ~(np.isnan(y_true_dep) | np.isnan(y_pred_dep))
        
        data_dict[key]['depression'].append({
            'run': run_num,
            'y_true': y_true_dep[valid_mask_dep].astype(int),
            'y_pred': y_pred_dep[valid_mask_dep].astype(int)
        })
        
        data_dict[key]['suicide_risk'].append({
            'run': run_num,
            'y_true': y_true_suicide,
            'y_pred': y_pred_suicide
        })
    
    return data_dict


def compute_metrics(y_true, y_pred, labels=None):
    """Compute performance metrics"""
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'per_class': {}
    }
    
    for i, label in enumerate(labels):
        metrics['per_class'][label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    # Macro and weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['macro_avg'] = {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1}
    metrics['weighted_avg'] = {'precision': weighted_p, 'recall': weighted_r, 'f1': weighted_f1}
    
    return metrics


def compute_confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix"""
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_cm(cm, title, save_path, labels):
    """Plot confusion matrix with Arial font"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 12, 'fontname': 'Arial'})
    plt.title(title, fontsize=14, fontweight='bold', fontname='Arial')
    plt.xlabel('Predicted Label', fontsize=12, fontname='Arial')
    plt.ylabel('True Label', fontsize=12, fontname='Arial')
    plt.xticks(fontname='Arial')
    plt.yticks(fontname='Arial')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_binary_task(data_dict, task_name, output_dir, label_names):
    """Analyze binary classification task"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {task_name}")
    print('='*60)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for key, runs in data_dict.items():
        all_y_true = []
        all_y_pred = []
        
        for run_data in runs:
            all_y_true.extend(run_data['y_true'])
            all_y_pred.extend(run_data['y_pred'])
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        labels = [0, 1]
        cm = compute_confusion_matrix(all_y_true, all_y_pred, labels)
        metrics = compute_metrics(all_y_true, all_y_pred, labels)
        
        results[key] = {
            'confusion_matrix': cm,
            'metrics': metrics,
            'total_samples': len(all_y_true)
        }
        
        # Plot confusion matrix
        title = f'{task_name} - {key}'
        save_path = os.path.join(output_dir, f'{key}_confusion_matrix.png')
        plot_cm(cm, title, save_path, label_names)
        
        # Print results
        print(f"\n{key}:")
        print(f"  Samples: {len(all_y_true)}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print(f"  No - P:{metrics['per_class'][0]['precision']:.4f}, R:{metrics['per_class'][0]['recall']:.4f}, F1:{metrics['per_class'][0]['f1']:.4f}")
        print(f"  Yes - P:{metrics['per_class'][1]['precision']:.4f}, R:{metrics['per_class'][1]['recall']:.4f}, F1:{metrics['per_class'][1]['f1']:.4f}")
        print(f"  Macro F1: {metrics['macro_avg']['f1']:.4f}")
    
    return results


def analyze_quad_task(data_dict, task_name, output_dir, class_labels, class_names):
    """Analyze 4-class classification task"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {task_name}")
    print('='*60)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for key, runs in data_dict.items():
        all_y_true = []
        all_y_pred = []
        
        for run_data in runs:
            all_y_true.extend(run_data['y_true'])
            all_y_pred.extend(run_data['y_pred'])
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        labels = [0, 1, 2, 3]
        cm = compute_confusion_matrix(all_y_true, all_y_pred, labels)
        metrics = compute_metrics(all_y_true, all_y_pred, labels)
        
        results[key] = {
            'confusion_matrix': cm,
            'metrics': metrics,
            'total_samples': len(all_y_true)
        }
        
        # Plot confusion matrix
        title = f'{task_name} - {key}'
        save_path = os.path.join(output_dir, f'{key}_confusion_matrix.png')
        plot_cm(cm, title, save_path, class_labels)
        
        # Print results
        print(f"\n{key}:")
        print(f"  Samples: {len(all_y_true)}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        for i, name in enumerate(class_names):
            m = metrics['per_class'][i]
            print(f"  {name} - P:{m['precision']:.4f}, R:{m['recall']:.4f}, F1:{m['f1']:.4f}, N:{int(m['support'])}")
        print(f"  Macro F1: {metrics['macro_avg']['f1']:.4f}")
    
    return results


def generate_report(results_dict, output_path):
    """Generate summary report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Classification Task Analysis Report\n\n")
        f.write("This report contains analysis results for four classification tasks:\n")
        f.write("1. Depression Binary (Yes/No)\n")
        f.write("2. Depression 4-Class (None/Mild/Moderate/Severe)\n")
        f.write("3. Suicide Risk Binary (Yes/No)\n")
        f.write("4. Suicide Risk 4-Class (No Risk/Low Risk/Medium Risk/High Risk)\n\n")
        f.write("---\n\n")
        
        # 1. Depression Binary
        f.write("## 1. Depression Binary Classification\n\n")
        for dataset_name, results in results_dict['depression_binary'].items():
            f.write(f"### {dataset_name}\n\n")
            for key, result in results.items():
                model, rag = key.rsplit('_', 1)
                f.write(f"#### {model} ({rag})\n\n")
                
                cm = result['confusion_matrix']
                f.write("**Confusion Matrix:**\n\n")
                f.write("| | Pred:No | Pred:Yes |\n")
                f.write("|---|---|---|\n")
                f.write(f"| True:No | {cm[0][0]} | {cm[0][1]} |\n")
                f.write(f"| True:Yes | {cm[1][0]} | {cm[1][1]} |\n\n")
                
                m = result['metrics']
                f.write(f"**Accuracy:** {m['accuracy']:.4f}\n\n")
                f.write("**Per-Class Metrics:**\n\n")
                f.write("| Class | Precision | Recall | F1 | Support |\n")
                f.write("|---|---|---|---|---|\n")
                f.write(f"| No | {m['per_class'][0]['precision']:.4f} | {m['per_class'][0]['recall']:.4f} | {m['per_class'][0]['f1']:.4f} | {int(m['per_class'][0]['support'])} |\n")
                f.write(f"| Yes | {m['per_class'][1]['precision']:.4f} | {m['per_class'][1]['recall']:.4f} | {m['per_class'][1]['f1']:.4f} | {int(m['per_class'][1]['support'])} |\n")
                f.write(f"| Macro Avg | {m['macro_avg']['precision']:.4f} | {m['macro_avg']['recall']:.4f} | {m['macro_avg']['f1']:.4f} | - |\n")
                f.write(f"| Weighted Avg | {m['weighted_avg']['precision']:.4f} | {m['weighted_avg']['recall']:.4f} | {m['weighted_avg']['f1']:.4f} | - |\n\n")
        
        # 2. Depression 4-Class
        f.write("## 2. Depression 4-Class Classification\n\n")
        for dataset_name, results in results_dict['depression_quad'].items():
            f.write(f"### {dataset_name}\n\n")
            for key, result in results.items():
                model, rag = key.rsplit('_', 1)
                f.write(f"#### {model} ({rag})\n\n")
                
                cm = result['confusion_matrix']
                f.write("**Confusion Matrix:**\n\n")
                f.write("| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |\n")
                f.write("|---|---|---|---|---|\n")
                for i, name in enumerate(DEPRESSION_QUAD_NAMES):
                    f.write(f"| True:{name} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} | {cm[i][3]} |\n")
                f.write("\n")
                
                m = result['metrics']
                f.write(f"**Accuracy:** {m['accuracy']:.4f}\n\n")
                f.write("**Per-Class Metrics:**\n\n")
                f.write("| Class | Precision | Recall | F1 | Support |\n")
                f.write("|---|---|---|---|---|\n")
                for i, name in enumerate(DEPRESSION_QUAD_NAMES):
                    pc = m['per_class'][i]
                    f.write(f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {int(pc['support'])} |\n")
                f.write(f"| Macro Avg | {m['macro_avg']['precision']:.4f} | {m['macro_avg']['recall']:.4f} | {m['macro_avg']['f1']:.4f} | - |\n")
                f.write(f"| Weighted Avg | {m['weighted_avg']['precision']:.4f} | {m['weighted_avg']['recall']:.4f} | {m['weighted_avg']['f1']:.4f} | - |\n\n")
        
        # 3. Suicide Risk Binary
        f.write("## 3. Suicide Risk Binary Classification\n\n")
        for dataset_name, results in results_dict['suicide_binary'].items():
            f.write(f"### {dataset_name}\n\n")
            for key, result in results.items():
                model, rag = key.rsplit('_', 1)
                f.write(f"#### {model} ({rag})\n\n")
                
                cm = result['confusion_matrix']
                f.write("**Confusion Matrix:**\n\n")
                f.write("| | Pred:No Risk | Pred:Has Risk |\n")
                f.write("|---|---|---|\n")
                f.write(f"| True:No Risk | {cm[0][0]} | {cm[0][1]} |\n")
                f.write(f"| True:Has Risk | {cm[1][0]} | {cm[1][1]} |\n\n")
                
                m = result['metrics']
                f.write(f"**Accuracy:** {m['accuracy']:.4f}\n\n")
                f.write("**Per-Class Metrics:**\n\n")
                f.write("| Class | Precision | Recall | F1 | Support |\n")
                f.write("|---|---|---|---|---|\n")
                f.write(f"| No Risk | {m['per_class'][0]['precision']:.4f} | {m['per_class'][0]['recall']:.4f} | {m['per_class'][0]['f1']:.4f} | {int(m['per_class'][0]['support'])} |\n")
                f.write(f"| Has Risk | {m['per_class'][1]['precision']:.4f} | {m['per_class'][1]['recall']:.4f} | {m['per_class'][1]['f1']:.4f} | {int(m['per_class'][1]['support'])} |\n")
                f.write(f"| Macro Avg | {m['macro_avg']['precision']:.4f} | {m['macro_avg']['recall']:.4f} | {m['macro_avg']['f1']:.4f} | - |\n")
                f.write(f"| Weighted Avg | {m['weighted_avg']['precision']:.4f} | {m['weighted_avg']['recall']:.4f} | {m['weighted_avg']['f1']:.4f} | - |\n\n")
        
        # 4. Suicide Risk 4-Class
        f.write("## 4. Suicide Risk 4-Class Classification\n\n")
        for dataset_name, results in results_dict['suicide_quad'].items():
            f.write(f"### {dataset_name}\n\n")
            for key, result in results.items():
                model, rag = key.rsplit('_', 1)
                f.write(f"#### {model} ({rag})\n\n")
                
                cm = result['confusion_matrix']
                f.write("**Confusion Matrix:**\n\n")
                f.write("| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |\n")
                f.write("|---|---|---|---|---|\n")
                for i, name in enumerate(SUICIDE_QUAD_NAMES):
                    f.write(f"| True:{name} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} | {cm[i][3]} |\n")
                f.write("\n")
                
                m = result['metrics']
                f.write(f"**Accuracy:** {m['accuracy']:.4f}\n\n")
                f.write("**Per-Class Metrics:**\n\n")
                f.write("| Class | Precision | Recall | F1 | Support |\n")
                f.write("|---|---|---|---|---|\n")
                for i, name in enumerate(SUICIDE_QUAD_NAMES):
                    pc = m['per_class'][i]
                    f.write(f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {int(pc['support'])} |\n")
                f.write(f"| Macro Avg | {m['macro_avg']['precision']:.4f} | {m['macro_avg']['recall']:.4f} | {m['macro_avg']['f1']:.4f} | - |\n")
                f.write(f"| Weighted Avg | {m['weighted_avg']['precision']:.4f} | {m['weighted_avg']['recall']:.4f} | {m['weighted_avg']['f1']:.4f} | - |\n\n")


def main():
    """Main function"""
    # Use relative paths for GitHub release
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_path = os.path.join(base_dir, 'data', 'classification_tasks')
    output_base = os.path.join(base_dir, 'results', 'classification_analysis')
    
    os.makedirs(output_base, exist_ok=True)
    
    all_results = {
        'depression_binary': {},
        'depression_quad': {},
        'suicide_binary': {},
        'suicide_quad': {}
    }
    
    # ========== Haodf Dataset ==========
    print("\n" + "="*60)
    print("Haodf Dataset")
    print("="*60)
    
    # Load binary data
    binary_data = load_binary_data(data_path, 'haodf_binary')
    
    # Depression binary
    depression_data = {k: v['depression'] for k, v in binary_data.items()}
    all_results['depression_binary']['Haodf'] = analyze_binary_task(
        depression_data, 'Haodf-Depression Binary',
        os.path.join(output_base, 'haodf_depression_binary'),
        BINARY_LABELS
    )
    
    # Suicide risk binary
    suicide_binary_data = {k: v['suicide_risk'] for k, v in binary_data.items()}
    all_results['suicide_binary']['Haodf'] = analyze_binary_task(
        suicide_binary_data, 'Haodf-Suicide Risk Binary',
        os.path.join(output_base, 'haodf_suicide_binary'),
        ['No Risk', 'Has Risk']
    )
    
    # Load 4-class data
    quad_data = load_quad_data(data_path, 'haodf_quaternary')
    
    # Depression 4-class
    depression_quad_data = {k: v['depression'] for k, v in quad_data.items()}
    all_results['depression_quad']['Haodf'] = analyze_quad_task(
        depression_quad_data, 'Haodf-Depression 4-Class',
        os.path.join(output_base, 'haodf_depression_quaternary'),
        DEPRESSION_QUAD_LABELS, DEPRESSION_QUAD_NAMES
    )
    
    # Suicide risk 4-class
    suicide_quad_data = {k: v['suicide_risk'] for k, v in quad_data.items()}
    all_results['suicide_quad']['Haodf'] = analyze_quad_task(
        suicide_quad_data, 'Haodf-Suicide Risk 4-Class',
        os.path.join(output_base, 'haodf_suicide_quaternary'),
        SUICIDE_QUAD_LABELS, SUICIDE_QUAD_NAMES
    )
    
    # ========== Standard Case Dataset ==========
    print("\n" + "="*60)
    print("Standard Case Dataset")
    print("="*60)
    
    # Load binary data
    binary_data = load_binary_data(data_path, 'standard_binary')
    
    # Depression binary
    depression_data = {k: v['depression'] for k, v in binary_data.items()}
    all_results['depression_binary']['Standard'] = analyze_binary_task(
        depression_data, 'Standard-Depression Binary',
        os.path.join(output_base, 'standard_depression_binary'),
        BINARY_LABELS
    )
    
    # Suicide risk binary
    suicide_binary_data = {k: v['suicide_risk'] for k, v in binary_data.items()}
    all_results['suicide_binary']['Standard'] = analyze_binary_task(
        suicide_binary_data, 'Standard-Suicide Risk Binary',
        os.path.join(output_base, 'standard_suicide_binary'),
        ['No Risk', 'Has Risk']
    )
    
    # Load 4-class data
    quad_data = load_quad_data(data_path, 'standard_quaternary')
    
    # Depression 4-class
    depression_quad_data = {k: v['depression'] for k, v in quad_data.items()}
    all_results['depression_quad']['Standard'] = analyze_quad_task(
        depression_quad_data, 'Standard-Depression 4-Class',
        os.path.join(output_base, 'standard_depression_quaternary'),
        DEPRESSION_QUAD_LABELS, DEPRESSION_QUAD_NAMES
    )
    
    # Suicide risk 4-class
    suicide_quad_data = {k: v['suicide_risk'] for k, v in quad_data.items()}
    all_results['suicide_quad']['Standard'] = analyze_quad_task(
        suicide_quad_data, 'Standard-Suicide Risk 4-Class',
        os.path.join(output_base, 'standard_suicide_quaternary'),
        SUICIDE_QUAD_LABELS, SUICIDE_QUAD_NAMES
    )
    
    # Generate report
    report_path = os.path.join(output_base, 'classification_analysis_report.md')
    generate_report(all_results, report_path)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"Report saved to: {report_path}")
    print("="*80)


if __name__ == '__main__':
    main()
