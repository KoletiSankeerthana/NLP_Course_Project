"""
Embedding Factory Module
------------------------
Unified interface for generating all 8 embedding types for the legal dataset.

Author: Antigravity (AI Assistant)
"""

import os
import numpy as np
import pandas as pd
from src.embeddings.bow_tfidf import generate_bow, generate_tfidf
from src.embeddings.word2vec import train_word2vec, get_sentence_embedding
from src.embeddings.fasttext import train_fasttext
from src.experimental.doc2vec import prepare_tagged_data, train_doc2vec, get_doc_vector
from src.experimental.transformer_models import TransformerEmbedder

class EmbeddingFactory:
    """
    Factory class to handle different embedding techniques consistently.
    """
    def __init__(self, corpus: list):
        self.corpus = corpus
        self.tokenized_corpus = [doc.split() for doc in corpus]

    def get_bow(self, max_features=5000):
        print("Generating BoW...")
        return generate_bow(self.corpus, max_features)

    def get_tfidf(self, max_features=5000):
        print("Generating TF-IDF...")
        return generate_tfidf(self.corpus, max_features)

    def get_word2vec(self, sg=0, vector_size=100):
        mode = "Skip-gram" if sg == 1 else "CBOW"
        print(f"Training Word2Vec ({mode})...")
        model = train_word2vec(self.tokenized_corpus, vector_size=vector_size, sg=sg)
        embeddings = np.array([get_sentence_embedding(model, doc) for doc in self.tokenized_corpus])
        return embeddings, model

    def get_fasttext(self, vector_size=100):
        print("Training FastText...")
        model = train_fasttext(self.tokenized_corpus, vector_size=vector_size)
        embeddings = np.array([model.wv[doc].mean(axis=0) if len(doc) > 0 else np.zeros(vector_size) for doc in self.tokenized_corpus])
        return embeddings, model

    def get_doc2vec(self, dm=1, vector_size=100):
        mode = "PV-DM" if dm == 1 else "PV-DBOW"
        print(f"Training Doc2Vec ({mode})...")
        tagged_data = prepare_tagged_data(self.corpus)
        model = train_doc2vec(tagged_data, vector_size=vector_size, dm=dm)
        embeddings = np.array([model.dv[str(i)] for i in range(len(self.corpus))])
        return embeddings, model

    def get_transformer(self, model_name='all-MiniLM-L6-v2'):
        print(f"Generating Transformer embeddings ({model_name})...")
        embedder = TransformerEmbedder(model_name)
        embeddings = embedder.get_embeddings(self.corpus)
        return embeddings, embedder

if __name__ == "__main__":
    # Small test
    sample_corpus = ["The court finds the defendant guilty.", "Appeal was dismissed by the high court."]
    factory = EmbeddingFactory(sample_corpus)
    
    # Test TF-IDF
    X_tfidf, vec = factory.get_tfidf()
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Test Word2Vec
    X_w2v, m_w2v = factory.get_word2vec()
    print(f"Word2Vec shape: {X_w2v.shape}")
