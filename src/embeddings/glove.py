"""
Dense Embedding Module: GloVe
-----------------------------
Handles loading and inference for Global Vectors (GloVe).
GloVe (Global Vectors for Word Representation) is based on matrix factorization 
of local context windows and global co-occurrence statistics.

Author: Antigravity (AI Assistant)
"""

import numpy as np
import joblib
import os
from src.utils.config import EMBEDDING_MODELS_PATH

def load_glove_embeddings(file_path):
    """
    Loads pre-trained GloVe vectors from a text file into a dictionary.
    """
    embeddings_dict = {}
    if not os.path.exists(file_path):
        print(f"Warning: GloVe file not found at {file_path}")
        return embeddings_dict

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def get_glove_sentence_embedding(embeddings_dict, tokens, vector_size=100):
    """
    Averages GloVe vectors for a sentence.
    """
    vectors = [embeddings_dict[word] for word in tokens if word in embeddings_dict]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def save_glove_dict(embeddings_dict, filename):
    """
    Saves the GloVe dictionary as a pickle for faster future loading.
    """
    if not os.path.exists(EMBEDDING_MODELS_PATH):
        os.makedirs(EMBEDDING_MODELS_PATH)
    path = os.path.join(EMBEDDING_MODELS_PATH, filename)
    joblib.dump(embeddings_dict, path)
    return path
