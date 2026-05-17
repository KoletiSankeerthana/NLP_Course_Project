"""
Sparse Embedding Module: One-Hot, BoW, and TF-IDF
--------------------------------------------------
Implements sparse vectorization techniques using scikit-learn.
Specialized for high-dimensional, interpretable text representations.

Techniques:
1. One-Hot Encoding (Manual & CountVectorizer)
2. Bag of Words (CountVectorizer)
3. TF-IDF (TfidfVectorizer)

Author: Antigravity (AI Assistant)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.utils.config import VECTORIZERS_PATH

def generate_one_hot(corpus, max_features=5000):
    """
    Generates One-Hot Encoding vectors.
    Note: For text, this is often represented as a binary BoW matrix.
    """
    vectorizer = CountVectorizer(max_features=max_features, binary=True)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def generate_bow(corpus, max_features=5000):
    """
    Generates Bag-of-Words (Frequency) vectors.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def generate_tfidf(corpus, max_features=5000):
    """
    Generates TF-IDF (Importance) vectors.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def save_vectorizer(vectorizer, filename):
    """
    Saves a vectorizer to the professional storage path.
    """
    if not os.path.exists(VECTORIZERS_PATH):
        os.makedirs(VECTORIZERS_PATH)
    path = os.path.join(VECTORIZERS_PATH, filename)
    joblib.dump(vectorizer, path)
    return path

if __name__ == "__main__":
    # Test
    test_corpus = ["Court ordered the file.", "The judge dismissed the appeal."]
    X, vec = generate_tfidf(test_corpus)
    print(f"TF-IDF Shape: {X.shape}")
