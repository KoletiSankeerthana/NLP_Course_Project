"""
Data Loading Utility
--------------------
Centralized module to ensure a single source of truth for the legal dataset.
"""

import pandas as pd
import os
import streamlit as st
from src.utils.config import PROCESSED_DATA_PATH, PROCESSED_CSV, RAW_DATA_PATH, RAW_CSV

@st.cache_data
def load_main_dataset():
    """
    Loads the primary legal dataset.
    Prioritizes the fully processed dataset, falls back to raw data if necessary.
    Enforces the research-standard filtering (Accepted/Rejected only).
    """
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_CSV)
    raw_path = os.path.join(RAW_DATA_PATH, RAW_CSV)
    
    df = None
    # Try loading the processed file first
    if os.path.exists(processed_path):
        try:
            df = pd.read_csv(processed_path)
        except: pass
            
    # Fallback to raw data if processed missing or incomplete
    if df is None or len(df) < 1000:
        if os.path.exists(raw_path):
            try:
                df = pd.read_csv(raw_path)
            except: pass
    
    if df is not None:
        # APPLY RESEARCH FILTERING (The 42,342 logic)
        if 'label' in df.columns:
            df = df[df['label'].isin(['Accepted', 'Rejected'])]
        
        # Drop rows missing critical text
        text_cols = ['processed_text', 'judgement', 'case_info']
        for col in text_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
                break
        
        return df
            
    return None

def get_dataset_statistics(df):
    """
    Returns a dictionary of real dataset metrics.
    """
    if df is None:
        return {}
    
    stats = {
        "total_records": len(df),
        "columns": list(df.columns),
        "classes": df['label'].nunique() if 'label' in df.columns else 0,
        "missing_values": df.isnull().sum().sum()
    }
    return stats
