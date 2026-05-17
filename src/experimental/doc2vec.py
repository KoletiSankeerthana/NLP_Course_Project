"""
Dense Embedding Module: Doc2Vec
-------------------------------
Implements Doc2Vec (PV-DM and PV-DBOW) using the Gensim library.
Doc2Vec is specifically designed to represent entire documents as vectors.

Author: Antigravity (AI Assistant)
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd

def prepare_tagged_data(corpus: list):
    """
    Wraps each document in a TaggedDocument object for Gensim's Doc2Vec.
    """
    return [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(corpus)]

def train_doc2vec(tagged_data: list, vector_size: int = 100, window: int = 5, dm: int = 1):
    """
    Trains a Doc2Vec model.
    
    Args:
        dm (int): 1 for PV-DM (distributed memory), 0 for PV-DBOW (distributed bag of words).
    """
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=2,
        workers=4,
        dm=dm,
        epochs=20
    )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def get_doc_vector(model, doc_words: list):
    """
    Infers a vector for a new document.
    """
    return model.infer_vector(doc_words)

if __name__ == "__main__":
    test_corpus = ["This is a legal document.", "Another case file for review."]
    tagged_data = prepare_tagged_data(test_corpus)
    model = train_doc2vec(tagged_data)
    print(f"Doc2Vec Vector Size: {model.vector_size}")
    vec = get_doc_vector(model, "This is a test document".split())
    print(f"Inferred Vector Sample: {vec[:5]}")
