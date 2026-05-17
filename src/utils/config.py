"""
Configuration Module
--------------------
Centralized paths and constants for the NLP legal project.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')

# Embedding paths
EMBEDDING_MODELS_PATH = os.path.join(BASE_DIR, 'embeddings', 'trained')
VECTORIZERS_PATH = os.path.join(BASE_DIR, 'embeddings', 'vectorizers')

# Model paths
TRAINED_MODELS_PATH = os.path.join(BASE_DIR, 'models', 'trained')

# Output paths
FIGURES_PATH = os.path.join(BASE_DIR, 'outputs', 'figures')
FIGURES_CM_PATH = os.path.join(FIGURES_PATH, 'confusion_matrices')
FIGURES_EMB_PATH = os.path.join(FIGURES_PATH, 'embedding_plots')
FIGURES_COMP_PATH = os.path.join(FIGURES_PATH, 'comparison_charts')
RESULTS_PATH = os.path.join(BASE_DIR, 'outputs', 'results')

# Dataset filenames
RAW_CSV = "case_files_total.csv"
PROCESSED_CSV = "processed_legal_dataset.csv"
FALLBACK_CSV = "case_files_total.csv"
