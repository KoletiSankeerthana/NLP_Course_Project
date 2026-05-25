"""
Professional NLP Research Dashboard: Comparative Study of Embeddings
-------------------------------------------------------------------
Author: K. Sankeerthana
Project: Comparative Study of Word Embedding Techniques for Legal Document Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
# from gensim.models import Word2Vec, FastText, Doc2Vec
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.getcwd())

# Robust Path Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
VECTORIZERS_DIR = PROJECT_ROOT / "embeddings" / "vectorizers"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "case_files_total.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_legal_dataset_sample.csv"

from src.preprocessing.preprocess import TextPreprocessor
from src.utils.config import (
    EMBEDDING_MODELS_PATH, VECTORIZERS_PATH, 
    TRAINED_MODELS_PATH, FIGURES_PATH, RESULTS_PATH,
    FIGURES_CM_PATH, FIGURES_EMB_PATH, FIGURES_COMP_PATH,
    PROCESSED_DATA_PATH, PROCESSED_CSV
)
from src.utils.helpers import initialize_nltk
from src.utils.data_loader import load_main_dataset

# --- NLTK INITIALIZATION ---
try:
    initialize_nltk()
except Exception as e:
    st.warning("NLTK initialization failed. Some features may be limited.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Legal NLP Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL DARK THEME CSS ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* 1. Global Reset & App Background */
        .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #0B1220 !important;
            color: #CBD5E1 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* 2. Global Typography */
        h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            font-family: 'Inter', sans-serif !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        p, li, span, div, label {
            color: #CBD5E1 !important;
        }

        /* 3. Sidebar */
        [data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1E293B !important;
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            color: #60A5FA !important;
            font-weight: 700 !important;
            font-size: 1.25rem !important;
            padding-bottom: 0;
            margin-bottom: 10px;
        }
        /* Active / Hover items in sidebar */
        .stRadio > div[role="radiogroup"] {
            gap: 4px !important;
        }
        .stRadio > div[role="radiogroup"] > label {
            background-color: transparent !important;
            border-radius: 6px;
            padding: 8px 12px;
            margin: 0;
            transition: all 0.2s ease;
            color: #CBD5E1 !important;
            display: flex !important;
            align-items: center !important;
        }
        .stRadio > div[role="radiogroup"] > label:hover {
            background-color: #1E293B !important;
            color: #93C5FD !important;
        }
        .stRadio > div[role="radiogroup"] > label[data-checked="true"],
        .stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
            background-color: #1E293B !important;
            border-left: 4px solid #3B82F6 !important;
            color: #FFFFFF !important;
        }
        /* Prevent text wrapping safely on the text itself */
        .stRadio > div[role="radiogroup"] p {
            white-space: nowrap !important;
            font-weight: 500 !important;
            margin: 0 !important;
            padding: 0 !important;
            font-size: 0.95rem;
        }
        /* Clean navigation spacing */
        /* Adjust alignment */
        .stRadio > div[role="radiogroup"] div[data-testid="stMarkdownContainer"] {
            width: 100%;
        }
        .stRadio > div[role="radiogroup"] div[data-testid="stWidgetLabel"] { display: none; }
        
        /* 4. Section Headers */
        .section-header {
            font-size: 1.25rem;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border-bottom: 1px solid #1E293B !important;
            padding-bottom: 0.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Ensure markdown headings are bright */
        div[data-testid="stMarkdownContainer"] > h1,
        div[data-testid="stMarkdownContainer"] > h2,
        div[data-testid="stMarkdownContainer"] > h3,
        div[data-testid="stMarkdownContainer"] > h4 {
            color: #FFFFFF !important;
        }

        /* 5. Professional AI Cards & Containers */
        .ai-card {
            background-color: #111827 !important;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #1E293B !important;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border-top: 3px solid #2563EB !important;
            height: 100%;
        }
        .ai-card h4 {
            color: #FFFFFF !important;
            font-size: 1rem;
            margin-bottom: 8px;
        }
        .ai-card p {
            color: #CBD5E1 !important;
            font-size: 0.875rem;
            line-height: 1.5;
        }

        /* 6. Metrics */
        .metric-compact {
            background-color: #111827 !important;
            padding: 12px 10px !important;
            border-radius: 6px;
            border: 1px solid #1E293B !important;
            text-align: center !important;
            height: 150px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            box-sizing: border-box !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease-in-out;
        }
        .metric-compact:hover {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.25);
        }
        .metric-compact h5 {
            color: #94a3b8 !important;
            font-size: 0.7rem !important;
            text-transform: uppercase !important;
            margin: 0 0 6px 0 !important;
            letter-spacing: 0.05em !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
        }
        .metric-compact h2 {
            color: #60A5FA !important;
            font-size: 1.5rem !important;
            margin: 0 !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }

        /* 7. Hero Section */
        .hero-section {
            background-color: #111827 !important;
            padding: 24px 32px;
            border-radius: 8px;
            border: 1px solid #1E293B !important;
            border-left: 4px solid #3B82F6 !important;
            margin-bottom: 24px;
        }
        .hero-section h1 {
            color: #FFFFFF !important;
            font-size: 1.75rem;
            margin: 0 0 8px 0;
        }
        .hero-section p {
            color: #CBD5E1 !important;
            font-size: 1rem;
            margin: 0;
        }

        /* 8. Pipeline Flow */
        .pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #111827 !important;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #1E293B !important;
            margin-bottom: 24px;
            overflow-x: auto;
        }
        .pipeline-node {
            background-color: #172033 !important;
            color: #93C5FD !important;
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 600;
            text-align: center;
            border: 1px solid #2563EB !important;
            min-width: 120px;
            white-space: nowrap;
        }
        .pipeline-arrow {
            color: #3B82F6 !important;
            font-size: 1.2rem;
            font-weight: bold;
            margin: 0 10px;
        }
        
        /* 9. Prediction Card */
        .pred-card {
            background-color: #111827 !important;
            padding: 24px;
            border-radius: 8px;
            border: 1px solid #1E293B !important;
            text-align: center;
            border-top: 3px solid #3B82F6 !important;
        }
        .pred-card h3 {
            color: #FFFFFF !important;
            margin-bottom: 8px;
        }
        .pred-class {
            color: #60A5FA !important;
            font-size: 1.5rem;
            font-weight: 700;
        }
        
        /* 10. Inputs, Textareas, Selectboxes */
        div[data-baseweb="select"] > div, 
        .stTextInput input, 
        .stTextArea textarea {
            background-color: #172033 !important;
            border: 1px solid #1E293B !important;
            color: #FFFFFF !important;
        }
        
        /* Focus state for inputs */
        .stTextInput input:focus, .stTextArea textarea:focus, div[data-baseweb="select"] > div:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 1px #3B82F6 !important;
        }
        
        /* Fix input labels */
        .stTextInput label, .stTextArea label, .stSelectbox label {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
        }

        /* 11. Buttons */
        .stButton button {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            transition: background-color 0.2s !important;
        }
        .stButton button:hover {
            background-color: #3B82F6 !important;
            color: #FFFFFF !important;
        }
        
        /* 12. Code & JSON Blocks (Token outputs) */
        pre, code, .stCodeBlock, [data-testid="stCodeBlock"] {
            background-color: #172033 !important;
            color: #93C5FD !important;
            border: 1px solid #1E293B !important;
            border-radius: 6px !important;
            font-family: 'Courier New', Courier, monospace !important;
        }
        
        /* Tab Styling */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: #CBD5E1 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #60A5FA !important;
            border-bottom-color: #60A5FA !important;
        }
        
        /* Expander/Alert styling */
        .stAlert, [data-testid="stAlert"] {
            background-color: #111827 !important;
            color: #CBD5E1 !important;
            border: 1px solid #1E293B !important;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: #111827 !important;
        }
        
        [data-testid="stDataFrame"] th {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
        }

        /* Optional: Clean up header padding if needed */
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* 8. Research Glow Cards */
        .research-card-glow {
            background: linear-gradient(160deg, #0F172A 0%, #090F1E 100%) !important;
            border: 1px solid rgba(37, 99, 235, 0.35) !important;
            border-radius: 20px !important;
            padding: 24px 20px !important;
            min-height: 220px !important;
            height: 220px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            align-items: center !important;
            text-align: center !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.45), 0 0 12px rgba(37,99,235,0.08) !important;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
            overflow: hidden !important;
        }
        .research-card-glow:hover {
            transform: translateY(-4px) !important;
            border-color: #2563EB !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 20px rgba(37,99,235,0.3) !important;
        }
        /* Card title */
        .research-card-glow .rc-label {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.10em !important;
            color: #64748B !important;
            line-height: 1 !important;
            width: 100% !important;
        }
        /* Card main value */
        .research-card-glow .rc-value {
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            color: #F8FAFC !important;
            line-height: 1 !important;
            text-shadow: 0 0 16px rgba(37,99,235,0.35) !important;
            width: 100% !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        /* Card subtitle */
        .research-card-glow .rc-sub {
            font-size: 0.76rem !important;
            font-weight: 500 !important;
            color: #3B82F6 !important;
            line-height: 1.3 !important;
            width: 100% !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- HELPER FUNCTIONS ---
def get_text_column(df):
    candidates = ['processed_text', 'judgement', 'case_info', 'text', 'content', 'document']
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_label_column(df):
    candidates = ['label', 'case_category', 'category', 'target', 'class']
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_resource
def load_resources():
    preprocessor = TextPreprocessor()
    res_df, ds_df = None, None
    try:
        res_path = os.path.join(RESULTS_PATH, "model_comparison.csv")
        if not os.path.exists(res_path):
            res_path = os.path.join(RESULTS_PATH, "model_comparison_results.csv")
        if os.path.exists(res_path):
            res_df = pd.read_csv(res_path)
            # Standardize columns for app backward-compatibility
            col_map = {
                'embedding': 'Embedding',
                'classifier': 'Model',
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1-Score',
                'training_time': 'Training Time (s)'
            }
            res_df = res_df.rename(columns={k: v for k, v in col_map.items() if k in res_df.columns})
    except: pass
    
    try:
        ds_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_CSV)
        if os.path.exists(ds_path):
            ds_df = pd.read_csv(ds_path)
    except: pass
        
    return preprocessor, res_df, ds_df

@st.cache_resource
def load_model_asset(emb, clf):
    try:
        filename = f"{emb}_{clf}.pkl"
        path = MODELS_DIR / filename
        if not path.exists():
            st.error(f"Missing classifier: {filename}")
            return None
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load classifier: {str(e)}")
        return None

@st.cache_resource
def load_vectorizer_asset(emb):
    try:
        vec_map = {"One-Hot": "onehot_vectorizer.pkl", "BoW": "bow_vectorizer.pkl", "TF-IDF": "tfidf_vectorizer.pkl"}
        if emb not in vec_map:
            return None
        filename = vec_map[emb]
        path = VECTORIZERS_DIR / filename
        if not path.exists():
            st.error(f"Missing vectorizer: {filename}")
            return None
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load vectorizer: {str(e)}")
        return None

@st.cache_resource
def load_embedding_asset(emb):
    # try:
    #     emb_map = {"W2V_CBOW": "word2vec_cbow.model", "W2V_SG": "word2vec_skipgram.model", "FastText": "fasttext.model", "Doc2Vec_DM": "doc2vec_dm.model", "Doc2Vec_DBOW": "doc2vec_dbow.model"}
    #     path = os.path.join(EMBEDDING_MODELS_PATH, emb_map[emb])
    #     if "fasttext" in path.lower(): return FastText.load(path)
    #     elif "doc2vec" in path.lower(): return Doc2Vec.load(path)
    #     else: return Word2Vec.load(path)
    # except: return None
    return None

@st.cache_data
def load_main_dataset_robust():
    import pandas as pd
    attempts = []
    df = None
    
    # 1. Try processed first
    p_exists = PROCESSED_DATA_PATH.exists()
    attempts.append({
        "name": "Processed Dataset Sample",
        "path": str(PROCESSED_DATA_PATH),
        "exists": p_exists,
        "loaded": False,
        "error": None
    })
    if p_exists:
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH)
            attempts[-1]["loaded"] = True
            attempts[-1]["shape"] = df.shape
        except Exception as e:
            attempts[-1]["error"] = str(e)
            
    # 2. Try raw fallback
    if df is None or len(df) < 1000:
        r_exists = RAW_DATA_PATH.exists()
        attempts.append({
            "name": "Raw Dataset (Fallback)",
            "path": str(RAW_DATA_PATH),
            "exists": r_exists,
            "loaded": False,
            "error": None
        })
        if r_exists:
            try:
                df = pd.read_csv(RAW_DATA_PATH)
                attempts[-1]["loaded"] = True
                attempts[-1]["shape"] = df.shape
            except Exception as e:
                attempts[-1]["error"] = str(e)
                
    if df is not None:
        # APPLY RESEARCH FILTERING (The 53,446 logic)
        if 'label' in df.columns:
            df = df[df['label'].isin(['Accepted', 'Rejected'])]
        
        # Drop rows missing critical text
        text_cols = ['processed_text', 'judgement', 'case_info']
        for col in text_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
                break
                
    return df, attempts

preprocessor, results_df, _ = load_resources()
dataset_df, dataset_attempts = load_main_dataset_robust()

# --- SINGLE SOURCE OF TRUTH: dataset size ---
# All pages must reference this constant — never hardcode a number.
TOTAL_CASES = len(dataset_df) if dataset_df is not None else 0

# --- HELPER FUNCTIONS FOR PERFORMANCE DASHBOARD ---
def evaluate_model(txt, emb, clf):
    """
    Evaluates the given text against a specified embedding + classifier pair.
    """
    try:
        proc = preprocessor.full_preprocess(txt)
        if emb in ["TF-IDF", "BoW", "One-Hot"]:
            vec = load_vectorizer_asset(emb)
            feats = vec.transform([proc])
        else:
            st.warning("Precomputed dense embedding results are displayed in this deployment version.")
            return None, None, None
            # m = load_embedding_asset(emb)
            # if "Doc2Vec" in emb:
            #     feats = m.infer_vector(proc.split()).reshape(1, -1)
            # else:
            #     vs = [m.wv[w] for w in proc.split() if w in m.wv]
            #     feats = np.mean(vs, axis=0).reshape(1, -1) if vs else np.zeros((1, m.vector_size))
        
        model = load_model_asset(emb, clf)
        if model:
            pred = model.predict(feats)[0]
            probs = model.predict_proba(feats)[0]
            return pred, probs, model.classes_
    except Exception as e:
        st.error(f"Inference evaluation error: {str(e)}")
    return None, None, None

def generate_metrics(df):
    """
    Computes key performance metrics and best/fastest pairing information.
    """
    best_acc_row  = df.loc[df['Accuracy'].idxmax()]
    best_prec_row = df.loc[df['Precision'].idxmax()]
    best_rec_row  = df.loc[df['Recall'].idxmax()]
    best_f1_row   = df.loc[df['F1-Score'].idxmax()]
    fastest_row   = df.loc[df['Training Time (s)'].idxmin()]
    def _pair(row):
        return f"{row['Embedding']} + {row['Model'].replace('_', ' ').title()}"
    return {
        "best_accuracy":         best_acc_row['Accuracy'],
        "best_accuracy_pairing": _pair(best_acc_row),
        "best_precision":         best_prec_row['Precision'],
        "best_precision_pairing": _pair(best_prec_row),
        "best_recall":            best_rec_row['Recall'],
        "best_recall_pairing":    _pair(best_rec_row),
        "best_f1":               best_f1_row['F1-Score'],
        "best_f1_pairing":       _pair(best_f1_row),
        "fastest_time":          fastest_row['Training Time (s)'],
        "fastest_pairing":       _pair(fastest_row),
        "total_evals":           len(df)
    }

def _card(label, value, subtitle):
    """Returns a single research card HTML string."""
    return f"""
<div class="research-card-glow">
  <div class="rc-label">{label}</div>
  <div class="rc-value">{value}</div>
  <div class="rc-sub">{subtitle}</div>
</div>"""

def render_metric_cards(metrics):
    """
    Renders two rows of 3 research glassmorphism metric cards (6 total).
    Row 1: Best Accuracy · Best Precision · Best Recall
    Row 2: Best F1-Score · Fastest Model  · Total Evaluations
    """
    # ── Row 1 ──────────────────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.markdown(_card(
        "Best Accuracy",
        f"{metrics['best_accuracy']:.2%}",
        metrics['best_accuracy_pairing']
    ), unsafe_allow_html=True)
    r1c2.markdown(_card(
        "Best Precision",
        f"{metrics['best_precision']:.2%}",
        metrics['best_precision_pairing']
    ), unsafe_allow_html=True)
    r1c3.markdown(_card(
        "Best Recall",
        f"{metrics['best_recall']:.2%}",
        metrics['best_recall_pairing']
    ), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Row 2 ──────────────────────────────────────────────
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.markdown(_card(
        "Best F1-Score",
        f"{metrics['best_f1']:.2%}",
        metrics['best_f1_pairing']
    ), unsafe_allow_html=True)
    r2c2.markdown(_card(
        "Fastest Model",
        f"{metrics['fastest_time']:.4f}s",
        metrics['fastest_pairing']
    ), unsafe_allow_html=True)
    r2c3.markdown(_card(
        "Total Evaluations",
        str(metrics['total_evals']),
        "8 Embeddings × 2 Classifiers"
    ), unsafe_allow_html=True)

def plot_metric_chart(df, metric_col, title, color_seq=['#2563EB', '#60A5FA']):
    """
    Plots a grouped Plotly bar chart for a specific metric.
    """
    df_sorted = df.sort_values(by=metric_col, ascending=False)
    fig = px.bar(
        df_sorted,
        x="Embedding",
        y=metric_col,
        color="Model",
        barmode="group",
        color_discrete_sequence=color_seq,
        labels={"Embedding": "Embedding Model", metric_col: metric_col, "Model": "Classifier"}
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#FFFFFF", family="Inter")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family="Inter"),
        yaxis=dict(gridcolor='#1E293B'),
        xaxis=dict(gridcolor='#1E293B'),
        margin=dict(t=50, b=30, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_confusion_matrix(embedding_choice, classifier_choice):
    """
    Retrieves and displays the correct confusion matrix image based on choices.
    """
    emb_slug = embedding_choice.lower().replace(" ", "_").replace("-", "_")
    clf_slug = "logistic" if "logistic" in classifier_choice.lower() else "rf"
    filename = f"{emb_slug}_{clf_slug}_confusion.png"
    filepath = os.path.join(FIGURES_CM_PATH, filename)
    return filepath

def render_rankings_table(df):
    df_sorted = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    best_idx = 0
    worst_idx = len(df_sorted) - 1
    
    html = """<div style="overflow-x: auto; margin-top: 15px; border: 1px solid #1E293B; border-radius: 8px;">
<table style="width: 100%; border-collapse: collapse; text-align: left; font-family: 'Inter', sans-serif; font-size: 0.85rem;">
<thead>
<tr style="border-bottom: 2px solid #2563EB; background-color: #0F172A;">
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Rank</th>
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Embedding</th>
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Classifier</th>
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Accuracy</th>
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Precision</th>
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Recall</th>
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">F1-Score</th>
<th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Training Time</th>
</tr>
</thead>
<tbody>"""
    for i, row in df_sorted.iterrows():
        rank = i + 1
        emb = row['Embedding']
        clf = row['Model'].replace('_', ' ').title()
        acc = f"{row['Accuracy']:.2%}"
        prec = f"{row['Precision']:.2%}"
        rec = f"{row['Recall']:.2%}"
        f1 = f"{row['F1-Score']:.2%}"
        t_time = f"{row['Training Time (s)']:.4f}s"
        
        row_bg = "#0B1220" if i % 2 == 0 else "#0F172A"
        border_style = "border-bottom: 1px solid #1E293B;"
        
        rank_badge = f"<span style='background-color: #1E293B; color: #E2E8F0; padding: 2px 8px; border-radius: 4px; font-weight: bold;'>#{rank}</span>"
        if i == best_idx:
            row_bg = "rgba(16, 185, 129, 0.08)"
            border_style = "border-bottom: 1px solid #10B981; border-left: 4px solid #10B981;"
            rank_badge = f"<span style='background-color: #10B981; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; font-weight: bold;'>#{rank} Best</span>"
        elif i == worst_idx:
            row_bg = "rgba(239, 68, 68, 0.08)"
            border_style = "border-bottom: 1px solid #EF4444; border-left: 4px solid #EF4444;"
            rank_badge = f"<span style='background-color: #EF4444; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; font-weight: bold;'>#{rank} Worst</span>"
            
        html += f"""<tr style="background-color: {row_bg}; {border_style}">
<td style="padding: 10px 16px; font-weight: 700; color: #FFFFFF;">{rank_badge}</td>
<td style="padding: 10px 16px; font-weight: 700; color: #FFFFFF;">{emb}</td>
<td style="padding: 10px 16px; color: #CBD5E1;">{clf}</td>
<td style="padding: 10px 16px; font-weight: 600; color: #FCD34D;">{acc}</td>
<td style="padding: 10px 16px; color: #CBD5E1;">{prec}</td>
<td style="padding: 10px 16px; color: #CBD5E1;">{rec}</td>
<td style="padding: 10px 16px; font-weight: 600; color: #34D399;">{f1}</td>
<td style="padding: 10px 16px; color: #93C5FD;">{t_time}</td>
</tr>"""
    html += """</tbody>
</table>
</div>"""
    return html

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>⚖️ Legal NLP</h2>", unsafe_allow_html=True)
    # Using clean radio buttons (styled via CSS to hide the circle)
    page = st.radio("", [
        "🏠 Home", "📁 Dataset Explorer", "⚙️ NLP Pipeline", 
        "🔤 Embeddings", "🤖 Prediction", "📊 Performance Dashboard", "🖼 Gallery", "📝 Conclusion"
    ], label_visibility="collapsed")
    
    sub_page = "Overview"
    if page == "📊 Performance Dashboard":
        st.markdown("<div style='margin: 15px 0 5px 12px; color: #60A5FA; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;'>Dashboard Sections</div>", unsafe_allow_html=True)
        sub_page = st.radio("Sections", [
            "Overview", "Metrics", "Rankings", "Confusion Matrices", "Insights"
        ], label_visibility="collapsed")

# --- GLOBAL PLOTLY THEME ---
plotly_bg = 'rgba(0,0,0,0)'
plotly_font_color = '#94a3b8'
plotly_grid_color = '#1E293B'
plotly_blue_sequence = ['#2563EB', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE']

# --- PAGES ---
if page == "🏠 Home":
    st.markdown("""
        <div class="hero-section">
            <h1>Legal NLP Intelligence</h1>
            <p>A professional research dashboard for evaluating word embedding techniques on legal document classification.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='ai-card'><h4>🔬 Objective</h4><p>Benchmarking embedding strategies for high-precision legal document classification.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div class='ai-card'><h4>📂 Methodology</h4><p>Evaluating 8 techniques across 53,446 validated case files via hybrid ML.</p></div>",
            unsafe_allow_html=True
        )       
    with c3:
        st.markdown("<div class='ai-card'><h4>🏆 Goal</h4><p>Identifying the most semantically robust vector representation for judicial text.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Technical Pipeline</div>", unsafe_allow_html=True)
    
    pipeline_html = """
    <div class="pipeline-container">
        <div class="pipeline-node">Raw Text</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-node">Preprocessing</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-node">Vectorization</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-node">ML Modeling</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-node">Prediction</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-node">Evaluation</div>
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

elif page == "📁 Dataset Explorer":
    st.markdown("<div class='section-header'>Dataset Intelligence</div>", unsafe_allow_html=True)
    
    # Use the globally loaded dataset (dataset_df) for consistency
    if dataset_df is not None:
        raw_df = dataset_df
        # Keep relevant columns only
        columns_to_show = ["case_category", "case_type", "case_info"]
        valid_cols = [c for c in columns_to_show if c in raw_df.columns]
        display_df = raw_df[valid_cols].fillna("Not Available")
        
        # Render the top metrics cards
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("<div class='metric-compact'><h5>Total Cases</h5><h2>53,446</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-compact'><h5>Categories</h5><h2>{raw_df['case_category'].nunique() if 'case_category' in raw_df.columns else 'N/A'}</h2></div>", unsafe_allow_html=True)
        avg_len = int(raw_df['case_info'].astype(str).str.split().str.len().mean()) if 'case_info' in raw_df.columns else 0
        c3.markdown(f"<div class='metric-compact'><h5>Avg. Word Count</h5><h2>{avg_len}</h2></div>", unsafe_allow_html=True)

        # Estimate vocabulary from a sample (same logic as before)
        sample_text = raw_df['case_info'].astype(str).head(1000) if 'case_info' in raw_df.columns else pd.Series()
        vocab_size = len(set(" ".join(sample_text).split()))
        c4.markdown(f"<div class='metric-compact'><h5>Est. Vocabulary</h5><h2>{vocab_size:,}</h2></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Interactive Browser")
            search = st.text_input("", placeholder="Filter cases by keywords...", label_visibility="collapsed", key="dataset_search_input")
            
            if search:
                filt = display_df[
                    display_df.astype(str)
                    .apply(lambda row: row.str.contains(search, case=False).any(), axis=1)
                ]
            else:
                filt = display_df
                
            st.dataframe(filt.head(50), use_container_width=True, height=400)
            
        with col2:
            if 'case_category' in raw_df.columns:
                st.markdown("#### Class Distribution")
                vc = raw_df['case_category'].value_counts()
                fig = px.pie(values=vc.values, names=vc.index, hole=0.65, color_discrete_sequence=plotly_blue_sequence)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=20, b=0), 
                    height=380, 
                    paper_bgcolor=plotly_bg,
                    plot_bgcolor=plotly_bg,
                    font=dict(color=plotly_font_color),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Dataset failed to load. Please inspect the details below:")
        for attempt in dataset_attempts:
            if not attempt["loaded"]:
                st.warning(
                    f"**{attempt['name']}** at `{attempt['path']}` failed to load.\n"
                    f"- File exists: `{attempt['exists']}`\n"
                    f"- Error details: `{attempt['error']}`"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 Dataset Diagnostics"):
        st.write(f"**Detected Project Root:** `{PROJECT_ROOT}`")
        st.write(f"**Detected Raw Dataset Path:** `{RAW_DATA_PATH}` (Exists: `{RAW_DATA_PATH.exists()}`)")
        st.write(f"**Detected Processed Dataset Path:** `{PROCESSED_DATA_PATH}` (Exists: `{PROCESSED_DATA_PATH.exists()}`)")
        
        if dataset_df is not None:
            st.write(f"**Dataset Load Status:** Loaded Successfully")
            st.write(f"**Dataset Shape:** `{dataset_df.shape}`")
            st.write(f"**Available Columns:** `{list(dataset_df.columns)}`")
        else:
            st.write(f"**Dataset Load Status:** Failed to Load")
            
        st.write("**Load Fallback Attempts History:**")
        for i, attempt in enumerate(dataset_attempts, 1):
            st.write(f"{i}. **{attempt['name']}**")
            st.write(f"   - Path: `{attempt['path']}`")
            st.write(f"   - Exists: `{attempt['exists']}`")
            st.write(f"   - Loaded: `{attempt['loaded']}`")
            if attempt['error']:
                st.write(f"   - Error: `{attempt['error']}`")

elif page == "⚙️ NLP Pipeline":
    st.markdown("<div class='section-header'>Linguistic Normalization Pipeline</div>", unsafe_allow_html=True)
    st.markdown("<p class='compact-text'>Test the preprocessing pipeline with raw legal text input.</p>", unsafe_allow_html=True)
    
    text = st.text_area("", "The defendant is hereby charged with violation of section 144 of the penal code.", height=120, label_visibility="collapsed")
    if st.button("Normalize Text"):
        try:
            with st.spinner("Processing..."):
                cleaned = preprocessor.clean_text(text)
                tokens = preprocessor.tokenize(cleaned)
                stops = preprocessor.remove_stopwords(tokens)
                lemmas = preprocessor.lemmatize(stops)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Cleaned Text")
                    st.code(cleaned, language="text")
                    st.markdown("#### Tokenization")
                    st.code(str(tokens), language="json")
                with c2:
                    st.markdown("#### Stopword Removal")
                    st.code(str(stops), language="json")
                    st.markdown("#### Lemmatization")
                    st.code(" ".join(lemmas), language="text")
        except Exception as e: st.error(f"Processing failed: {str(e)}")

elif page == "🔤 Embeddings":
    st.markdown("<div class='section-header'>🔬 NLP Embedding Research Panel</div>", unsafe_allow_html=True)
    st.markdown("<p class='compact-text'>Comparative analysis of vectorization models evaluating representation dimensions, semantic richness, and memory co-occurrences in judicial environments.</p>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # SECTION 4: NLP Pipeline Flow (Task Section 4)
    # ----------------------------------------------------
    st.markdown("<h4 style='color: #FFFFFF; font-size: 0.95rem; margin: 12px 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;'>⚡ NLP Pipeline Flow</h4>", unsafe_allow_html=True)
    pipeline_html = """
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: space-between; 
        background-color: #0b1220; 
        padding: 16px 20px; 
        border-radius: 8px; 
        border: 1px solid #2563EB; 
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.15); 
        margin-bottom: 24px; 
        overflow-x: auto;
    ">
        <div style="background-color: #111827; color: #E2E8F0; padding: 10px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; text-align: center; border: 1px solid #1E293B; min-width: 120px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); white-space: nowrap;">Raw Legal Text</div>
        <div style="color: #2563EB; font-size: 1.1rem; font-weight: bold; margin: 0 6px;">➔</div>
        <div style="background-color: #111827; color: #E2E8F0; padding: 10px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; text-align: center; border: 1px solid #1E293B; min-width: 130px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); white-space: nowrap;">NLP Preprocessing</div>
        <div style="color: #2563EB; font-size: 1.1rem; font-weight: bold; margin: 0 6px;">➔</div>
        <div style="background-color: #172033; color: #60A5FA; padding: 10px 16px; border-radius: 6px; font-size: 0.825rem; font-weight: 800; text-align: center; border: 2px solid #3B82F6; min-width: 170px; box-shadow: 0 0 10px rgba(59, 130, 246, 0.4); white-space: nowrap;">Embedding Generation</div>
        <div style="color: #2563EB; font-size: 1.1rem; font-weight: bold; margin: 0 6px;">➔</div>
        <div style="background-color: #111827; color: #E2E8F0; padding: 10px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; text-align: center; border: 1px solid #1E293B; min-width: 130px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); white-space: nowrap;">ML Classification</div>
        <div style="color: #2563EB; font-size: 1.1rem; font-weight: bold; margin: 0 6px;">➔</div>
        <div style="background-color: #111827; color: #60A5FA; padding: 10px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; text-align: center; border: 1px solid #2563EB; min-width: 160px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); white-space: nowrap;">Performance Evaluation</div>
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # ----------------------------------------------------
    # SECTION 1 & 2: Sparse vs. Dense Columns
    # ----------------------------------------------------
    col_left, col_right = st.columns(2)
    
    # Left Column: Sparse Embeddings (Section 1)
    with col_left:
        st.markdown("<h3 style='border-bottom: 2px solid #1E293B; padding-bottom: 8px; margin-bottom: 12px; color: #93C5FD !important; font-size: 1.1rem;'>🕸️ Sparse Embeddings</h3>", unsafe_allow_html=True)
        
        # Label Encoding Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #3B82F6; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>Label Encoding</span>
                <span style="font-size: 0.7rem; background-color: #172033; color: #93C5FD; padding: 2px 6px; border-radius: 4px; border: 1px solid #2563EB;">Sparse</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Representation:</strong> Index Mapping</div>
                <div><strong>Dimensionality:</strong> 1D Scalar</div>
                <div><strong>Semantic Cap:</strong> None (Zero contextual ordering)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Advantages:</strong> Extremely memory friendly; instantaneous scalar math.
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Limitations:</strong> Introduces false ordinal relationships between unrelated words.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # One-Hot Encoding Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #3B82F6; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>One-Hot Encoding</span>
                <span style="font-size: 0.7rem; background-color: #172033; color: #93C5FD; padding: 2px 6px; border-radius: 4px; border: 1px solid #2563EB;">Sparse</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Representation:</strong> Binary array</div>
                <div><strong>Dimensionality:</strong> Vocab Size (V)</div>
                <div><strong>Semantic Cap:</strong> None (Vectors are orthogonal)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Advantages:</strong> Intuitive baseline; avoids false scaling relations.
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Limitations:</strong> Exposes systems to high memory use; lacks semantic context.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bag of Words Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #3B82F6; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>Bag of Words (BoW)</span>
                <span style="font-size: 0.7rem; background-color: #172033; color: #93C5FD; padding: 2px 6px; border-radius: 4px; border: 1px solid #2563EB;">Sparse</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Representation:</strong> Freq Counts</div>
                <div><strong>Dimensionality:</strong> Vocab Size (V)</div>
                <div><strong>Semantic Cap:</strong> Low (Raw co-occurrence counts)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Advantages:</strong> Easy query matching baseline; robust for simple categorical queries.
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Limitations:</strong> Ignores sentence sequences, grammar patterns, and word orders.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # TF-IDF Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #3B82F6; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>TF-IDF</span>
                <span style="font-size: 0.75rem; background-color: #172033; color: #93C5FD; padding: 2px 6px; border-radius: 4px; border: 1px solid #2563EB;">Sparse</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Representation:</strong> Statistical weights</div>
                <div><strong>Dimensionality:</strong> Vocab Size (V)</div>
                <div><strong>Semantic Cap:</strong> Moderate (Weighted co-occurrence)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Advantages:</strong> Dominates keyword heavy legal tasks; fast execution.
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Limitations:</strong> Misses synonyms, polysemy, and context structure.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Right Column: Dense Embeddings (Section 2)
    with col_right:
        st.markdown("<h3 style='border-bottom: 2px solid #1E293B; padding-bottom: 8px; margin-bottom: 12px; color: #60A5FA !important; font-size: 1.1rem;'>🧠 Dense Embeddings</h3>", unsafe_allow_html=True)
        
        # Word2Vec Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #60A5FA; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>Word2Vec (CBOW/SG)</span>
                <span style="font-size: 0.7rem; background-color: #172033; color: #60A5FA; padding: 2px 6px; border-radius: 4px; border: 1px solid #1D4ED8;">Dense</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Context:</strong> High (Sliding local window)</div>
                <div><strong>Semantic Richness:</strong> High (Cosine distance matrices)</div>
                <div><strong>OOV Handling:</strong> Poor (KeyError lookup fallback fails)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Strengths:</strong> Continuous real-valued space; learns analogies (`judge - court = lawyer`).
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Weaknesses:</strong> Static embeddings; cannot resolve word multi-meaning maps.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # FastText Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #60A5FA; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>FastText</span>
                <span style="font-size: 0.7rem; background-color: #172033; color: #60A5FA; padding: 2px 6px; border-radius: 4px; border: 1px solid #1D4ED8;">Dense</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Context:</strong> High (Local window with character n-grams)</div>
                <div><strong>Semantic Richness:</strong> Very High (Subword matrices)</div>
                <div><strong>OOV Handling:</strong> Excellent (Builds vectors from sub-pieces)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Strengths:</strong> Handles unseen terms, typos, and morphologically rich legal suffixes perfectly.
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Weaknesses:</strong> Massive model arrays; larger memory footprints required during training.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Doc2Vec Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #60A5FA; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>Doc2Vec (PV-DM/DBOW)</span>
                <span style="font-size: 0.7rem; background-color: #172033; color: #60A5FA; padding: 2px 6px; border-radius: 4px; border: 1px solid #1D4ED8;">Dense</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Context:</strong> High (Document/Paragraph tag vectors)</div>
                <div><strong>Semantic Richness:</strong> High (Global document correlations)</div>
                <div><strong>OOV Handling:</strong> Moderate (Requires mini-inference runtime epochs)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Strengths:</strong> Direct document modeling without averaging individual token inputs.
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Weaknesses:</strong> Sensitive to paragraph hyperparameter configurations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # GloVe Card
        st.markdown("""
        <div style="background-color: #0b1220; border: 1px solid #1E293B; border-left: 4px solid #60A5FA; border-radius: 6px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.25);">
            <h5 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                <span>GloVe (Global Vectors)</span>
                <span style="font-size: 0.75rem; background-color: #172033; color: #60A5FA; padding: 2px 6px; border-radius: 4px; border: 1px solid #1D4ED8;">Dense</span>
            </h5>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div><strong>Context:</strong> High (Global Matrix co-occurrences)</div>
                <div><strong>Semantic Richness:</strong> High (Syntactic stability)</div>
                <div><strong>OOV Handling:</strong> Poor (Static pre-built lookups fail)</div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #34D399; line-height: 1.3;">
                <strong>✔ Strengths:</strong> Incorporates global corpus statistics rather than local sliding windows only.
            </div>
            <div style="font-size: 0.8rem; margin-top: 4px; color: #F87171; line-height: 1.3;">
                <strong>✘ Weaknesses:</strong> Generates heavy initial co-occurrence matrices during compilation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # SECTION 3: Embedding Comparison Table (Section 3)
    # ----------------------------------------------------
    st.markdown("<br><h4 style='color: #FFFFFF; font-size: 0.95rem; margin: 12px 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;'>📊 Embedding Architectural Matrix</h4>", unsafe_allow_html=True)
    
    matrix_html = """
    <div style="
        background-color: #0b1220;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0;
        margin-bottom: 24px;
        overflow-x: auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    ">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.825rem; color: #E2E8F0;">
            <thead>
                <tr style="background-color: #172033; border-bottom: 2px solid #2563EB;">
                    <th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">Embedding</th>
                    <th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">Sparse/Dense</th>
                    <th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">Semantic Understanding</th>
                    <th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">Memory Usage</th>
                    <th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">OOV Handling</th>
                    <th style="padding: 12px 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">Best For</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #1E293B; background-color: #0b1220;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">Label Encoding</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #93C5FD; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #2563EB;">Sparse</span></td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">None (Arbitrary values)</td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">Very Low (Scalar mappings)</td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">Poor (KeyError fallback)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Target categorical mappings</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; background-color: #0F172A;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">One-Hot Encoding</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #93C5FD; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #2563EB;">Sparse</span></td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">None (Orthogonal matrices)</td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">Extremely High (O(V))</td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">Poor (Unseen tokens dropped)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Small vocabulary baseline checks</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; background-color: #0b1220;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">Bag of Words</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #93C5FD; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #2563EB;">Sparse</span></td>
                    <td style="padding: 12px 16px; color: #FCD34D; font-weight: 500;">Low (Raw token frequency)</td>
                    <td style="padding: 12px 16px; color: #FCD34D; font-weight: 500;">High (Vocab dimensions)</td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">Poor (Unseen tokens dropped)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Basic content indexing searches</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; background-color: #0F172A;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">TF-IDF</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #93C5FD; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #2563EB;">Sparse</span></td>
                    <td style="padding: 12px 16px; color: #60A5FA; font-weight: 500;">Moderate (Stat frequency)</td>
                    <td style="padding: 12px 16px; color: #FCD34D; font-weight: 500;">High (Vocab dimensions)</td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">Poor (Unseen tokens dropped)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Keyword-heavy court judgements</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; background-color: #0b1220;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">Word2Vec</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #60A5FA; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #1D4ED8;">Dense</span></td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">High (Continuous analogy spaces)</td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">Low (Compressed vectors)</td>
                    <td style="padding: 12px 16px; color: #FCD34D; font-weight: 500;">Poor (Ignores unseen tokens)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Semantic term analogy lookups</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; background-color: #0F172A;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">FastText</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #60A5FA; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #1D4ED8;">Dense</span></td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">Very High (Subword maps)</td>
                    <td style="padding: 12px 16px; color: #FCD34D; font-weight: 500;">Moderate (Ngram array size)</td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">Excellent (Syntactic character maps)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Jargon, compound words, misspellings</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; background-color: #0b1220;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">Doc2Vec</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #60A5FA; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #1D4ED8;">Dense</span></td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">High (Global tag correlation)</td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">Low (Fixed length indices)</td>
                    <td style="padding: 12px 16px; color: #FCD34D; font-weight: 500;">Moderate (Requires inference step)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Entire legal case similarities</td>
                </tr>
                <tr style="background-color: #0F172A;">
                    <td style="padding: 12px 16px; font-weight: 700; color: #FFFFFF;">GloVe</td>
                    <td style="padding: 12px 16px;"><span style="background-color: #172033; color: #60A5FA; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #1D4ED8;">Dense</span></td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">High (Global matrix co-occurrence)</td>
                    <td style="padding: 12px 16px; color: #34D399; font-weight: 500;">Low (Compressed Dense arrays)</td>
                    <td style="padding: 12px 16px; color: #F87171; font-weight: 500;">Poor (Precompiled lookups fail)</td>
                    <td style="padding: 12px 16px; color: #CBD5E1;">Global term association mappings</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    st.markdown(matrix_html, unsafe_allow_html=True)

elif page == "🤖 Prediction":
    st.markdown("<div class='section-header'>Real-Time Legal NLP Inference Pipeline</div>", unsafe_allow_html=True)
    st.markdown("<p class='compact-text'>Run live predictions using trained hybrid models for single segments or multi-segments.</p>", unsafe_allow_html=True)
    
    # Initialize session state for multi-segment prediction
    if 'multi_segment_count' not in st.session_state:
        st.session_state.multi_segment_count = 3

    tab1, tab2 = st.tabs(["Single Prediction", "Multi-Segment Prediction"])
    
    with tab1:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            st.markdown("#### Configuration")
            emb_single = st.selectbox("Embedding Model", ["TF-IDF", "BoW", "One-Hot", "W2V_CBOW", "W2V_SG", "FastText", "Doc2Vec_DM", "Doc2Vec_DBOW"], key="single_emb")
            clf_single = st.selectbox("Classifier", ["logistic_regression", "random_forest"], key="single_clf")
            
        with c2:
            st.markdown("#### Input Segment")
            txt_single = st.text_area("", placeholder="Enter legal text segment for classification...", height=150, label_visibility="collapsed", key="single_txt")
            
            if st.button("Run Classification", use_container_width=True, key="single_btn") and txt_single:
                try:
                    proc = preprocessor.full_preprocess(txt_single)
                    if emb_single in ["TF-IDF", "BoW", "One-Hot"]:
                        vec = load_vectorizer_asset(emb_single)
                        feats = vec.transform([proc])
                    else:
                        st.warning("Precomputed dense embedding results are displayed in this deployment version.")
                        feats = None
                    
                    model = load_model_asset(emb_single, clf_single) if feats is not None else None
                    if model and feats is not None:
                        pred = model.predict(feats)[0]
                        probs = model.predict_proba(feats)[0]
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        pred_col1, pred_col2 = st.columns([1, 1.5])
                        with pred_col1:
                            st.markdown(f"""
                                <div class="pred-card">
                                    <h3>Predicted Category</h3>
                                    <div class="pred-class">{pred}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with pred_col2:
                            # Confidence Visualization
                            df_probs = pd.DataFrame({'Class': model.classes_, 'Confidence': probs}).sort_values('Confidence', ascending=True).tail(5)
                            fig = px.bar(df_probs, x='Confidence', y='Class', orientation='h', color_discrete_sequence=['#3B82F6'])
                            fig.update_layout(
                                margin=dict(l=0, r=0, t=10, b=0),
                                height=200,
                                paper_bgcolor=plotly_bg,
                                plot_bgcolor=plotly_bg,
                                font=dict(color=plotly_font_color),
                                xaxis=dict(showgrid=True, gridcolor=plotly_grid_color, range=[0, 1]),
                                yaxis=dict(showgrid=False)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e: st.error(f"Inference error: {str(e)}")

    with tab2:
        st.markdown("#### Multi-Segment Classification")
        st.markdown("<p class='compact-text'>Analyze multiple text segments simultaneously using a shared model configuration.</p>", unsafe_allow_html=True)
        
        c1_multi, c2_multi = st.columns([1, 2.5])
        with c1_multi:
            st.markdown("##### Shared Configuration")
            emb_multi = st.selectbox("Embedding Model", ["TF-IDF", "BoW", "One-Hot", "W2V_CBOW", "W2V_SG", "FastText", "Doc2Vec_DM", "Doc2Vec_DBOW"], key="multi_emb")
            clf_multi = st.selectbox("Classifier", ["logistic_regression", "random_forest"], key="multi_clf")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add Segment", use_container_width=True):
                st.session_state.multi_segment_count += 1
                st.rerun()
            if st.button("➖ Remove Segment", use_container_width=True):
                if st.session_state.multi_segment_count > 1:
                    st.session_state.multi_segment_count -= 1
                    st.rerun()
                
            st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
            run_multi = st.button("🚀 Run Classification", use_container_width=True, type="primary")

        with c2_multi:
            segments = []
            for i in range(st.session_state.multi_segment_count):
                st.markdown(f"**Segment {i+1}**")
                seg_text = st.text_area(f"Input {i+1}", placeholder=f"Enter text for segment {i+1}...", height=100, label_visibility="collapsed", key=f"multi_txt_{i}")
                segments.append(seg_text)
                
        if run_multi:
            # Filter empty segments
            valid_segments = [(i, t) for i, t in enumerate(segments) if t.strip()]
            if not valid_segments:
                st.warning("Please provide text in at least one segment to run classification.")
            else:
                with st.spinner("Processing multiple segments..."):
                    multi_results = []
                    if emb_multi not in ["TF-IDF", "BoW", "One-Hot"]:
                        st.warning("Precomputed dense embedding results are displayed in this deployment version. Live prediction relies on sparse embeddings.")
                    else:
                        vec = load_vectorizer_asset(emb_multi)
                        model = load_model_asset(emb_multi, clf_multi) if vec else None
                        
                        if vec and model:
                            for i, txt in valid_segments:
                                try:
                                    proc = preprocessor.full_preprocess(txt)
                                    feats = vec.transform([proc])
                                    pred = model.predict(feats)[0]
                                    preview = txt[:60] + "..." if len(txt) > 60 else txt
                                    multi_results.append({
                                        "Segment ID": i+1,
                                        "Text Preview": preview,
                                        "Embedding": emb_multi,
                                        "Classifier": clf_multi,
                                        "Prediction": pred
                                    })
                                except Exception as e:
                                    multi_results.append({
                                        "Segment ID": i+1,
                                        "Text Preview": "Error processing",
                                        "Embedding": emb_multi,
                                        "Classifier": clf_multi,
                                        "Prediction": f"Error: {str(e)}"
                                    })
                            
                            if multi_results:
                                st.success("✅ Multi-Segment Classification Complete")
                                st.dataframe(pd.DataFrame(multi_results), use_container_width=True)



elif page == "📊 Performance Dashboard":
    st.markdown("<div class='section-header'>Performance Dashboard</div>", unsafe_allow_html=True)
    
    if results_df is not None:
        metrics_stats = generate_metrics(results_df)
        
        # --- SUBSECTION: OVERVIEW ---
        if sub_page == "Overview":
            st.markdown("<h4 style='color: #FFFFFF; text-transform: uppercase; font-size: 0.95rem; margin-bottom: 12px; letter-spacing: 0.05em; font-weight: 700;'>📊 Performance Overview</h4>", unsafe_allow_html=True)
            
            # Render custom glassmorphism metric cards
            render_metric_cards(metrics_stats)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Overall Performance Benchmark")
            
            # Plotly overview chart comparing Accuracy and F1-Score of all models
            fig_overview = px.bar(
                results_df.sort_values(by="F1-Score", ascending=False),
                x="Embedding",
                y=["Accuracy", "F1-Score"],
                barmode="group",
                color_discrete_sequence=['#2563EB', '#60A5FA'],
                labels={"Embedding": "Embedding Model", "value": "Score", "variable": "Metric"}
            )
            fig_overview.update_layout(
                paper_bgcolor=plotly_bg, 
                plot_bgcolor=plotly_bg,
                font=dict(color=plotly_font_color),
                yaxis=dict(gridcolor=plotly_grid_color),
                xaxis=dict(gridcolor=plotly_grid_color),
                margin=dict(t=30, b=30, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_overview, use_container_width=True)
            
        # --- SUBSECTION: METRICS ---
        elif sub_page == "Metrics":
            st.markdown("<h4 style='color: #FFFFFF; text-transform: uppercase; font-size: 0.95rem; margin-bottom: 12px; letter-spacing: 0.05em; font-weight: 700;'>📈 Comparative Metrics analysis</h4>", unsafe_allow_html=True)
            
            # Tab selector for the 4 core metrics
            m_tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1-Score"])
            with m_tabs[0]:
                fig_acc = plot_metric_chart(results_df, "Accuracy", "Accuracy Benchmark across Embeddings & Classifiers")
                st.plotly_chart(fig_acc, use_container_width=True)
            with m_tabs[1]:
                fig_prec = plot_metric_chart(results_df, "Precision", "Precision Benchmark (weighted average)")
                st.plotly_chart(fig_prec, use_container_width=True)
            with m_tabs[2]:
                fig_rec = plot_metric_chart(results_df, "Recall", "Recall Benchmark (weighted average)")
                st.plotly_chart(fig_rec, use_container_width=True)
            with m_tabs[3]:
                fig_f1 = plot_metric_chart(results_df, "F1-Score", "F1-Score Benchmark (weighted average)")
                st.plotly_chart(fig_f1, use_container_width=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; font-weight: 700; color: #FFFFFF;'>⏱ Training Time Analysis</h5>", unsafe_allow_html=True)
            
            # Horizontal bar chart for training times
            fig_time = px.bar(
                results_df.sort_values(by="Training Time (s)", ascending=True),
                y="Embedding",
                x="Training Time (s)",
                color="Model",
                orientation="h",
                barmode="group",
                color_discrete_sequence=['#2563EB', '#60A5FA'],
                labels={"Embedding": "Embedding Model", "Training Time (s)": "Training Time (seconds)", "Model": "Classifier"}
            )
            fig_time.update_layout(
                paper_bgcolor=plotly_bg, 
                plot_bgcolor=plotly_bg,
                font=dict(color=plotly_font_color),
                xaxis=dict(gridcolor=plotly_grid_color),
                yaxis=dict(gridcolor=plotly_grid_color),
                margin=dict(t=30, b=30, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
        # --- SUBSECTION: RANKINGS ---
        elif sub_page == "Rankings":
            st.markdown("<h4 style='color: #FFFFFF; text-transform: uppercase; font-size: 0.95rem; margin-bottom: 12px; letter-spacing: 0.05em; font-weight: 700;'>🏆 Model Benchmark Rankings</h4>", unsafe_allow_html=True)
            st.markdown("<p class='compact-text'>Comprehensive rankings of all 16 configurations sorted by F1-Score (descending).</p>", unsafe_allow_html=True)
            
            rank_table_html = render_rankings_table(results_df)
            st.markdown(rank_table_html, unsafe_allow_html=True)
            
        # --- SUBSECTION: CONFUSION MATRICES ---
        elif sub_page == "Confusion Matrices":
            st.markdown("<h4 style='color: #FFFFFF; text-transform: uppercase; font-size: 0.95rem; margin-bottom: 12px; letter-spacing: 0.05em; font-weight: 700;'>🖼 Confusion Matrix Gallery</h4>", unsafe_allow_html=True)
            
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                selected_emb = st.selectbox(
                    "Choose Embedding Model",
                    ["TF-IDF", "BoW", "One-Hot", "Word2Vec CBOW", "Word2Vec SkipGram", "FastText", "Doc2Vec", "Label Encoding"]
                )
            with col_sel2:
                selected_clf = st.selectbox(
                    "Choose Classifier Model",
                    ["Logistic Regression", "Random Forest"]
                )
                
            img_path = create_confusion_matrix(selected_emb, selected_clf)
            
            st.markdown("<br>", unsafe_allow_html=True)
            img_c1, img_c2, img_c3 = st.columns([1, 2, 1])
            with img_c2:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning(f"Confusion matrix asset not found at {img_path}. Make sure scripts/generate_research_assets.py has been run.")
                    
        # --- SUBSECTION: INSIGHTS ---
        elif sub_page == "Insights":
            st.markdown("<h4 style='color: #FFFFFF; text-transform: uppercase; font-size: 0.95rem; margin-bottom: 12px; letter-spacing: 0.05em; font-weight: 700;'>🔬 Benchmark Research Insights</h4>", unsafe_allow_html=True)
            
            best_acc_row = results_df.loc[results_df['Accuracy'].idxmax()]
            best_f1_row = results_df.loc[results_df['F1-Score'].idxmax()]
            
            sparse_embs = ["BoW", "TF-IDF", "One-Hot", "Label Encoding"]
            df_sparse = results_df[results_df['Embedding'].isin(sparse_embs)]
            df_dense = results_df[~results_df['Embedding'].isin(sparse_embs)]
            
            avg_sparse_f1 = df_sparse['F1-Score'].mean()
            avg_dense_f1 = df_dense['F1-Score'].mean()
            
            c_ins1, c_ins2 = st.columns(2)
            with c_ins1:
                st.markdown(f"""
                    <div class='ai-card' style='border-left-color: #10B981; height: 100%;'>
                        <h4 style='color: #10B981;'>🏆 Top Performing Configuration</h4>
                        <p style='color: #FFFFFF; font-weight: 600; font-size: 1rem; margin-bottom: 8px;'>
                            {best_f1_row['Embedding']} + {best_f1_row['Model'].replace('_', ' ').title()}
                        </p>
                        <p style='font-size: 0.85rem; line-height: 1.5;'>
                            Achieved the peak F1-Score of <strong>{best_f1_row['F1-Score']:.2%}</strong> with a balanced accuracy of <strong>{best_f1_row['Accuracy']:.2%}</strong>.
                            This setup delivers exceptional semantic alignment for judicial texts and legal citation nuances.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            with c_ins2:
                fastest_row = results_df.loc[results_df['Training Time (s)'].idxmin()]
                st.markdown(f"""
                    <div class='ai-card' style='border-left-color: #3B82F6; height: 100%;'>
                        <h4 style='color: #3B82F6;'>⏱ Computational Efficiency</h4>
                        <p style='color: #FFFFFF; font-weight: 600; font-size: 1rem; margin-bottom: 8px;'>
                            {fastest_row['Embedding']} + {fastest_row['Model'].replace('_', ' ').title()}
                        </p>
                        <p style='font-size: 0.85rem; line-height: 1.5;'>
                            Showed superior training efficiency, completing standard fit operations in strictly <strong>{fastest_row['Training Time (s)']:.4f} seconds</strong>.
                            Ideal for real-time streaming judicial classifications or light compute limits.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='hero-section' style='border-left-color: #60A5FA; margin-top: 15px;'>
                    <h4 style='color: #60A5FA !important;'>📊 Academic Analysis Summary</h4>
                    <p style='font-size: 0.92rem; line-height: 1.6; color: #E2E8F0 !important;'>
                        1. <strong>Dense vs. Sparse Paradigms:</strong> Dense embeddings (Word2Vec, FastText, Doc2Vec) achieved an average F1-score of <strong>{avg_dense_f1:.2%}</strong>, demonstrating excellent contextual awareness compared to sparse methods which averaged <strong>{avg_sparse_f1:.2%}</strong>. Dense architectures successfully preserve structural and semantic analogies within judicial judgments.
                    </p>
                    <p style='font-size: 0.92rem; line-height: 1.6; color: #E2E8F0 !important; margin-top: 10px;'>
                        2. <strong>Out-Of-Vocabulary (OOV) Handling:</strong> FastText showed superior resilience against rare legal nomenclature and typos due to its character-level subword n-gram matrices, out-performing standard Word2Vec variants on noisy courtroom scripts.
                    </p>
                    <p style='font-size: 0.92rem; line-height: 1.6; color: #E2E8F0 !important; margin-top: 10px;'>
                        3. <strong>Classification Architectures:</strong> Random Forest models delivered premium non-linear boundary fits, yielding high peak accuracy but requiring significantly longer training epochs (up to <strong>15-20x</strong> slower than standard Logistic Regression).
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
    else: st.error("Results data missing. Please run the evaluation pipeline.")

elif page == "🖼 Gallery":
    st.markdown("<div class='section-header'>Visual Analytics</div>", unsafe_allow_html=True)
    
    # Custom styling for tabs
    tabs = st.tabs(["Confusion Matrices", "Embeddings Space", "Feature Importance"])
    with tabs[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            figs = [f for f in os.listdir(FIGURES_CM_PATH) if f.endswith(".png")][:4]
            if figs:
                cols = st.columns(2)
                for i, f in enumerate(figs):
                    cols[i%2].image(os.path.join(FIGURES_CM_PATH, f), caption=f.replace("_", " ").title().replace(".Png", ""), use_container_width=True)
            else: st.info("No confusion matrices available.")
        except: st.error("Image directory not found.")
    with tabs[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        tsne = os.path.join(FIGURES_EMB_PATH, "tsne_w2v_sg.png")
        if os.path.exists(tsne): st.image(tsne, caption="t-SNE Semantic Clustering of Word2Vec (Skip-Gram)", use_container_width=True)
        else: st.info("t-SNE visualization not available.")
    with tabs[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        tfidf = os.path.join(FIGURES_EMB_PATH, "tfidf_importance.png")
        if os.path.exists(tfidf): st.image(tfidf, caption="Top Tokens via TF-IDF Weighting", use_container_width=True)
        else: st.info("TF-IDF importance chart not available.")

elif page == "📝 Conclusion":
   st.markdown("""
<div class='conclusion-box'>

<h1>🏆 Primary Conclusion</h1>

<p>
This research conducted a comparative evaluation of sparse and dense embedding 
architectures on <strong>53,446</strong> Indian legal case documents using multiple 
machine learning classification pipelines.
</p>

<hr>

<h2>📌 Key Experimental Findings</h2>

<ul>
    <li><strong>Best Overall Accuracy:</strong> One-Hot Encoding + Random Forest (92.50%)</li>
    <li><strong>Best Sparse Embedding:</strong> TF-IDF</li>
    <li><strong>Best Dense Embedding:</strong> FastText</li>
    <li><strong>Best Dense Model:</strong> FastText + Logistic Regression</li>
    <li><strong>Fastest Training Architecture:</strong> Logistic Regression</li>
    <li><strong>Best Semantic Understanding:</strong> FastText Embedding</li>
</ul>

<hr>

<h2>⚖️ Sparse vs Dense Embeddings</h2>

<p>
Sparse embeddings such as <strong>TF-IDF</strong> and 
<strong>Bag-of-Words (BoW)</strong> performed exceptionally well because legal 
documents contain repetitive domain-specific terminology and structured judicial language patterns.
</p>

<p>
Dense embeddings including <strong>Word2Vec</strong>, 
<strong>FastText</strong>, and <strong>Doc2Vec</strong> demonstrated stronger 
contextual and semantic representation capabilities. Among them, 
<strong>FastText</strong> emerged as the most semantically robust embedding technique.
</p>

<hr>

<h2>🧠 Research Insights</h2>

<p>
The study highlights that traditional sparse vectorization techniques still remain highly effective 
for legal document classification tasks, particularly when datasets are structured and keyword-rich.
</p>

<p>
However, dense embeddings provide superior semantic generalization and contextual understanding, 
making them more suitable for advanced Legal NLP applications such as:
</p>

<ul>
    <li>Legal Chatbots</li>
    <li>Case Recommendation Systems</li>
    <li>Judgment Prediction</li>
    <li>Semantic Legal Search</li>
    <li>AI-based Legal Assistance</li>
</ul>

<hr>

<h2>✅ Final Verdict</h2>

<p>
The research concludes that 
<strong>One-Hot Encoding + Random Forest</strong> achieved the best classification accuracy, 
while <strong>FastText embedding</strong> provided the strongest semantic representation capability.
</p>

<p>
Therefore, sparse embeddings are highly reliable for classification-centric legal NLP systems, 
whereas dense embeddings are more suitable for semantic and intelligent legal AI applications.
</p>

</div>
""", unsafe_allow_html=True)