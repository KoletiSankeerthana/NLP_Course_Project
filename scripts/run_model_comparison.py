"""
Experiment Runner: Step 4 - Model Training & Evaluation
------------------------------------------------------
Trains Logistic Regression and Random Forest classifiers across all 
embedding techniques and generates a consolidated comparison report.

Author: Antigravity (AI Assistant)
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import time
from sklearn.model_selection import train_test_split
#from gensim.models import Word2Vec, FastText, Doc2Vec


# Ensure src is in path
sys.path.append(os.getcwd())

from src.utils.config import (
    PROCESSED_DATA_PATH, PROCESSED_CSV, 
    VECTORIZERS_PATH, EMBEDDING_MODELS_PATH, 
    RESULTS_PATH, TRAINED_MODELS_PATH
)
from src.models.train_model import ModelTrainer
from src.evaluation.evaluate import compute_metrics, plot_confusion_matrix, plot_comparison_metrics
from src.utils.helpers import setup_logger, ensure_dir

# Initialize logger
logger = setup_logger("ModelEvaluation")

def generate_dense_features(corpus, model, vector_size=100):
    """Generates averaged document embeddings."""
    features = []
    for doc in corpus:
        tokens = doc.split()
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if not vectors:
            features.append(np.zeros(vector_size))
        else:
            features.append(np.mean(vectors, axis=0))
    return np.array(features)

def run_step_4_modeling():
    # 1. Load Data
    data_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_CSV)
    df = pd.read_csv(data_path).dropna(subset=['processed_text', 'case_category'])
    
    X_raw = df['processed_text'].astype(str)
    y = df['case_category']
    labels = sorted(y.unique())
    
    # Consistent Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    
    results = []
    classifiers = ['logistic_regression', 'random_forest']
    ensure_dir(TRAINED_MODELS_PATH)
    ensure_dir(RESULTS_PATH)

    # --- Sparse Embeddings ---
    sparse_map = {
        "BoW": "bow_vectorizer.pkl",
        "TF-IDF": "tfidf_vectorizer.pkl",
        "One-Hot": "onehot_vectorizer.pkl"
    }
    
    for emb_name, vec_file in sparse_map.items():
        logger.info(f"Evaluating Sparse Embedding: {emb_name}")
        vectorizer = joblib.load(os.path.join(VECTORIZERS_PATH, vec_file))
        X_train = vectorizer.transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        
        for clf_name in classifiers:
            trainer = ModelTrainer(clf_name)
            t_start = time.time()
            trainer.train(X_train, y_train)
            train_time = time.time() - t_start
            
            y_pred = trainer.model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, clf_name, emb_name, train_time)
            results.append(metrics)
            trainer.save_model(emb_name)
            plot_confusion_matrix(y_test, y_pred, labels, f"{emb_name} {clf_name}")

    # --- Dense Embeddings ---
    dense_map = {
        "W2V_CBOW": "word2vec_cbow.model",
        "W2V_SG": "word2vec_skipgram.model",
        "FastText": "fasttext.model",
        "Doc2Vec_DM": "doc2vec_dm.model",
        "Doc2Vec_DBOW": "doc2vec_dbow.model"
    }
    
    for emb_name, mod_file in dense_map.items():
        logger.info(f"Evaluating Dense Embedding: {emb_name}")
        mod_path = os.path.join(EMBEDDING_MODELS_PATH, mod_file)
        
        if "fasttext" in mod_file: model = FastText.load(mod_path)
        elif "doc2vec" in mod_file: model = Doc2Vec.load(mod_path)
        else: model = Word2Vec.load(mod_path)
            
        if "doc2vec" in mod_file:
            X_train = np.array([model.infer_vector(doc.split()) for doc in X_train_raw])
            X_test = np.array([model.infer_vector(doc.split()) for doc in X_test_raw])
        else:
            X_train = generate_dense_features(X_train_raw, model)
            X_test = generate_dense_features(X_test_raw, model)
            
        for clf_name in classifiers:
            trainer = ModelTrainer(clf_name)
            t_start = time.time()
            trainer.train(X_train, y_train)
            train_time = time.time() - t_start
            
            y_pred = trainer.model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, clf_name, emb_name, train_time)
            results.append(metrics)
            trainer.save_model(emb_name)
            plot_confusion_matrix(y_test, y_pred, labels, f"{emb_name} {clf_name}")

    # Save and Plot
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "model_comparison_results.csv"), index=False)
    plot_comparison_metrics(results_df, metric='Accuracy')
    plot_comparison_metrics(results_df, metric='F1-Score')
    
    logger.info("Step 4 Completed. Results saved to outputs/results/")

if __name__ == "__main__":
    run_step_4_modeling()
