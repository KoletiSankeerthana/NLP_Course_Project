"""
Dense Embedding Module: FastText
--------------------------------
Implements FastText (subword-level) embeddings using Gensim.
FastText excels at handling Out-of-Vocabulary (OOV) words by learning 
representations for n-grams of characters.

Author: Antigravity (AI Assistant)
"""

from gensim.models import FastText
import numpy as np
import os
from src.utils.config import EMBEDDING_MODELS_PATH

def train_fasttext(sentences, vector_size=100, window=5, min_count=2, workers=4):
    """
    Trains a FastText model.
    """
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model

def save_fasttext_model(model, filename):
    """
    Serializes the FastText model.
    """
    if not os.path.exists(EMBEDDING_MODELS_PATH):
        os.makedirs(EMBEDDING_MODELS_PATH)
    path = os.path.join(EMBEDDING_MODELS_PATH, filename)
    model.save(path)
    return path
