"""
Visualization Utility Module
----------------------------
Provides functions for generating linguistic and statistical 
visualizations of the legal dataset.

Author: Antigravity (AI Assistant)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from src.utils.config import FIGURES_PATH

def plot_class_distribution(df, column='case_category'):
    """Plots the frequency of each legal category."""
    plt.figure(figsize=(10, 6))
    sns.countplot(y=column, data=df, palette='viridis')
    plt.title('Distribution of Legal Case Categories')
    plt.xlabel('Number of Documents')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'comparison_charts/class_distribution.png'))
    plt.close()

def plot_text_length_distribution(df, column='processed_text'):
    """Plots the distribution of document lengths."""
    df['text_len'] = df[column].str.split().str.len()
    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_len'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Document Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'comparison_charts/text_length_dist.png'))
    plt.close()

if __name__ == "__main__":
    print("Visualization utilities initialized.")
