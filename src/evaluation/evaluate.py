"""
Evaluation Module
-----------------
Provides research-quality evaluation metrics and comparative visualizations
for NLP model performance analysis.

Author: Antigravity (AI Assistant)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from src.utils.config import FIGURES_PATH

def compute_metrics(y_true, y_pred, model_name, embedding_name, train_time):
    """
    Computes a comprehensive suite of metrics for a model-embedding pair.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics = {
        'Embedding': embedding_name,
        'Model': model_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training Time (s)': train_time
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels, title_suffix):
    """
    Generates and saves a heatmap confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {title_suffix}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    filename = f"confusion_matrix_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(FIGURES_PATH, filename))
    plt.close()

def plot_comparison_metrics(results_df, metric='Accuracy'):
    """
    Creates a bar plot comparing a specific metric across all embeddings.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Embedding', y=metric, hue='Model', data=results_df, palette='viridis')
    plt.title(f'Comparative Analysis: {metric} across Embedding Techniques')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    filename = f"comparison_{metric.lower().replace('-', '_')}.png"
    plt.savefig(os.path.join(FIGURES_PATH, filename))
    plt.close()

if __name__ == "__main__":
    # Test
    print("Evaluation module loaded.")
