"""
Dense Embedding Module: Word2Vec
--------------------------------
Implements Word2Vec (CBOW and Skip-Gram) using the Gensim library.
Word2Vec learns dense, semantic representations through shallow neural networks.

Author: Antigravity (AI Assistant)
"""

from gensim.models import Word2Vec
import numpy as np
import os
from src.utils.config import EMBEDDING_MODELS_PATH

def train_word2vec(sentences, vector_size=100, window=5, min_count=2, sg=0, workers=4):
    """
    Trains a Word2Vec model.
    
    Args:
        sentences (list): List of tokenized sentences.
        sg (int): 0 for CBOW, 1 for Skip-gram.
    """
    model = Word2Vec(
        sentences=sentences, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=workers,
        sg=sg
    )
    return model

def get_sentence_embedding(model, tokens):
    """
    Computes an average vector for a sentence.
    """
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def save_w2v_model(model, filename):
    """
    Saves a Word2Vec model.
    """
    if not os.path.exists(EMBEDDING_MODELS_PATH):
        os.makedirs(EMBEDDING_MODELS_PATH)
    path = os.path.join(EMBEDDING_MODELS_PATH, filename)
    model.save(path)
    return path
