# =========================================================
# IMPORTS
# =========================================================

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================================================
# PROJECT ROOT FIX
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# =========================================================
# CUSTOM IMPORTS
# =========================================================

from src.preprocessing.preprocess import TextPreprocessor

# =========================================================
# PATHS
# =========================================================

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_legal_dataset_sample.csv"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "case_files_total.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "model_comparison_results.csv"
)

VECTORIZER_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "vectorizers"
)

TFIDF_PATH = VECTORIZER_DIR / "tfidf_vectorizer.pkl"
BOW_PATH = VECTORIZER_DIR / "bow_vectorizer.pkl"
ONEHOT_PATH = VECTORIZER_DIR / "onehot_vectorizer.pkl"

# =========================================================
# LOAD RESOURCES
# =========================================================

@st.cache_resource
def load_resources():

    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    results_df = None
    dataset_df = None

    # =====================================================
    # LOAD RESULTS
    # =====================================================

    try:

        if RESULTS_PATH.exists():

            results_df = pd.read_csv(RESULTS_PATH)

            # Standardize column names
            column_map = {
                "embedding": "Embedding",
                "classifier": "Model",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1-Score",
                "training_time": "Training Time (s)"
            }

            results_df.rename(
                columns={
                    k: v
                    for k, v in column_map.items()
                    if k in results_df.columns
                },
                inplace=True
            )

        else:

            st.warning(
                f"Results CSV not found:\n{RESULTS_PATH}"
            )

    except Exception as e:

        st.error(
            f"Error loading results CSV:\n{str(e)}"
        )

    # =====================================================
    # LOAD DATASET
    # =====================================================

    try:

        if PROCESSED_DATA_PATH.exists():

            dataset_df = pd.read_csv(
                PROCESSED_DATA_PATH
            )

        elif RAW_DATA_PATH.exists():

            dataset_df = pd.read_csv(
                RAW_DATA_PATH
            )

        else:

            st.warning(
                "Dataset files not found."
            )

    except Exception as e:

        st.error(
            f"Dataset loading failed:\n{str(e)}"
        )

    return preprocessor, results_df, dataset_df

# =========================================================
# LOAD EVERYTHING
# =========================================================

preprocessor, results_df, dataset_df = load_resources()

# =========================================================
# OPTIONAL DEBUG
# =========================================================

with st.expander("🔍 System Diagnostics"):

    st.write("PROJECT ROOT:", PROJECT_ROOT)

    st.write(
        "Processed Dataset Exists:",
        PROCESSED_DATA_PATH.exists()
    )

    st.write(
        "Raw Dataset Exists:",
        RAW_DATA_PATH.exists()
    )

    st.write(
        "Results CSV Exists:",
        RESULTS_PATH.exists()
    )

    st.write(
        "TFIDF Vectorizer Exists:",
        TFIDF_PATH.exists()
    )

    st.write(
        "BoW Vectorizer Exists:",
        BOW_PATH.exists()
    )

    st.write(
        "OneHot Vectorizer Exists:",
        ONEHOT_PATH.exists()
    )

    if dataset_df is not None:

        st.write(
            "Dataset Shape:",
            dataset_df.shape
        )

    if results_df is not None:

        st.write(
            "Results Shape:",
            results_df.shape
        )

# =========================================================
# SAFE GLOBAL VARIABLES
# =========================================================

TOTAL_CASES = (
    len(dataset_df)
    if dataset_df is not None
    else 0
)