# =========================================================
# IMPORTS
# =========================================================

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================================================
# PROJECT ROOT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add root to Python path
sys.path.append(str(PROJECT_ROOT))

# =========================================================
# CUSTOM IMPORTS
# =========================================================

from src.preprocessing.preprocess import TextPreprocessor

# =========================================================
# DATASET PATHS
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

# =========================================================
# RESULTS PATH
# =========================================================

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "model_comparison_results.csv"
)

# =========================================================
# VECTORIZER PATHS
# =========================================================

VECTORIZER_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "vectorizers"
)

TFIDF_PATH = VECTORIZER_DIR / "tfidf_vectorizer.pkl"
BOW_PATH = VECTORIZER_DIR / "bow_vectorizer.pkl"
ONEHOT_PATH = VECTORIZER_DIR / "onehot_vectorizer.pkl"

# =========================================================
# DEBUGGING INFO
# =========================================================

with st.expander("🔍 System Diagnostics"):

    st.write("PROJECT ROOT:", PROJECT_ROOT)

    st.write("Processed Dataset:", PROCESSED_DATA_PATH)
    st.write("Processed Exists:", PROCESSED_DATA_PATH.exists())

    st.write("Raw Dataset:", RAW_DATA_PATH)
    st.write("Raw Exists:", RAW_DATA_PATH.exists())

    st.write("Results Path:", RESULTS_PATH)
    st.write("Results Exists:", RESULTS_PATH.exists())

    st.write("TFIDF Exists:", TFIDF_PATH.exists())
    st.write("BoW Exists:", BOW_PATH.exists())
    st.write("OneHot Exists:", ONEHOT_PATH.exists())

# =========================================================
# LOAD RESOURCES
# =========================================================

@st.cache_resource
def load_resources():

    preprocessor = TextPreprocessor()

    results_df = None
    dataset_df = None

    # =====================================================
    # LOAD RESULTS CSV
    # =====================================================

    try:

        if RESULTS_PATH.exists():

            results_df = pd.read_csv(RESULTS_PATH)

            st.success("✅ Results CSV Loaded Successfully")

            # Standardize columns
            col_map = {
                'embedding': 'Embedding',
                'classifier': 'Model',
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1-Score',
                'training_time': 'Training Time (s)'
            }

            results_df = results_df.rename(
                columns={
                    k: v
                    for k, v in col_map.items()
                    if k in results_df.columns
                }
            )

            st.write("Results Shape:", results_df.shape)

        else:

            st.error(
                f"❌ Missing results file:\n{RESULTS_PATH}"
            )

    except Exception as e:

        st.error(
            f"❌ Error loading results CSV:\n{str(e)}"
        )

    # =====================================================
    # LOAD DATASET
    # =====================================================

    try:

        if PROCESSED_DATA_PATH.exists():

            dataset_df = pd.read_csv(PROCESSED_DATA_PATH)

            st.success("✅ Processed Dataset Loaded Successfully")

        elif RAW_DATA_PATH.exists():

            dataset_df = pd.read_csv(RAW_DATA_PATH)

            st.success("✅ Raw Dataset Loaded Successfully")

        else:

            st.error("❌ No dataset files found.")

        if dataset_df is not None:

            st.write("Dataset Shape:", dataset_df.shape)
            st.write("Dataset Columns:", list(dataset_df.columns))

    except Exception as e:

        st.error(
            f"❌ Dataset loading failed:\n{str(e)}"
        )

    return preprocessor, results_df, dataset_df

# =========================================================
# LOAD EVERYTHING
# =========================================================

preprocessor, results_df, dataset_df = load_resources()

# =========================================================
# TOTAL CASES
# =========================================================

TOTAL_CASES = (
    len(dataset_df)
    if dataset_df is not None
    else 0
)

# =========================================================
# METRICS
# =========================================================

c1, c2, c3 = st.columns(3)

c1.markdown(
    f"""
    <div class='metric-compact'>
        <h5>Total Cases</h5>
        <h2>{TOTAL_CASES:,}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SAFE RESULTS CHECK
# =========================================================

if results_df is None:

    st.error(
        "❌ Results data missing. Please check results CSV."
    )

else:

    st.success(
        "✅ Performance Dashboard Loaded Successfully"
    )

    st.dataframe(results_df.head())

# =========================================================
# SAFE VECTORIZER CHECK
# =========================================================

if not TFIDF_PATH.exists():
    st.error(f"❌ Missing vectorizer: {TFIDF_PATH}")

if not BOW_PATH.exists():
    st.error(f"❌ Missing vectorizer: {BOW_PATH}")

if not ONEHOT_PATH.exists():
    st.error(f"❌ Missing vectorizer: {ONEHOT_PATH}")