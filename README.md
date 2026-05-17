# ⚖️ Legal NLP Intelligence System
## Comparative Study of Sparse and Dense Embedding Architectures for Indian Legal Document Classification

---

# 📌 Project Overview

The **Legal NLP Intelligence System** is an advanced Natural Language Processing research project designed to analyze, preprocess, vectorize, classify, and evaluate Indian legal case documents using both **Sparse** and **Dense Embedding Architectures**.

This project performs a comprehensive comparative study between traditional vectorization techniques and semantic embedding models for judicial text classification.

The system integrates:
- NLP preprocessing pipelines
- Sparse & Dense embeddings
- Machine Learning classifiers
- Experimental benchmarking
- Interactive Streamlit dashboard
- Real-time legal text prediction

---

# 🎯 Research Objective

The primary objective of this research is to evaluate the effectiveness of different text embedding techniques for legal document intelligence systems.

The study focuses on:
- Legal document preprocessing
- Embedding representation learning
- Classification performance comparison
- Semantic representation quality
- Sparse vs Dense embedding analysis

---

# 📂 Dataset Information

### Dataset Used
Indian Legal Case Documents Dataset

### Dataset Size
**53,446** legal case documents

### Categories
- Civil Cases
- Criminal Cases

### Dataset Files
```bash
data/raw/case_files_total.csv
````

### Processed Dataset

```bash
data/processed/processed_legal_dataset_sample.csv
```

---

# ⚙️ NLP Preprocessing Pipeline

The project includes a complete legal text preprocessing pipeline.

### Preprocessing Operations

* Lowercasing
* Special character cleaning
* Numerical artifact removal
* Tokenization
* Stopword removal
* Stemming
* Lemmatization
* POS tagging
* Text normalization

---

# 🧠 NLP Pipeline Architecture

```text
Raw Legal Text
       ↓
Text Cleaning
       ↓
Normalization
       ↓
Tokenization
       ↓
Stopword Removal
       ↓
Stemming & Lemmatization
       ↓
Embedding Generation
       ↓
Machine Learning Classification
       ↓
Evaluation & Visualization
```

---

# 📊 Embedding Techniques Evaluated

## Sparse Embeddings

1. Label-encoding
2. One-Hot Encoding
3. Bag-of-Words (BoW)
4. TF-IDF 

## Dense Embeddings

5. Word2Vec CBOW
6. Word2Vec Skip-Gram
7. GLOVE
8. FastText

---

# 🤖 Machine Learning Classifiers

The following classifiers were used:

1. Logistic Regression
2. Random Forest

### Total Experimental Evaluations

```text
8 Embeddings × 2 Classifiers = 16 Evaluations
```

---

# 📈 Evaluation Metrics

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Training Time
* Semantic Representation Capability

---

# 🏆 Research Findings

## ✅ Best Overall Accuracy

### One-Hot Encoding + Random Forest

**Accuracy Achieved:** 92.50%

---

## ✅ Best Sparse Embedding

### TF-IDF

TF-IDF demonstrated excellent keyword discrimination for structured legal terminology.

---

## ✅ Best Dense Embedding

### FastText

FastText achieved superior semantic understanding and contextual representation.

---

## ✅ Best Dense Model

### FastText + Logistic Regression

---

## ✅ Fastest Training Architecture

### Logistic Regression

---

# 📌 Key Research Insights

### Sparse Embeddings

Sparse embeddings performed exceptionally well because legal documents contain repetitive domain-specific terminology and structured judicial patterns.

### Dense Embeddings

Dense embeddings demonstrated stronger contextual and semantic understanding capabilities.

### Why FastText Performed Best

FastText:

* handles rare legal terms
* captures subword information
* improves semantic similarity
* generalizes unseen words effectively

---

# 🚀 Features of the System

## 🏠 Home Dashboard

* Project overview
* Research objectives
* Technical workflow
* Experimental summary

---

## 📂 Dataset Explorer

* Interactive dataset browsing
* Dataset statistics
* Search/filter functionality
* Class distribution visualization

---

## ⚙️ NLP Pipeline

* Live preprocessing visualization
* Text normalization
* Tokenization
* Lemmatization outputs
* NLP transformation stages

---

## 🧠 Embedding Analysis

* Sparse vs Dense comparison
* Embedding architecture visualization
* Embedding performance benchmarking

---

## 🤖 AI Prediction Engine

* Real-time legal document classification
* Dynamic embedding selection
* Classifier selection
* Confidence prediction

---

## 📊 Performance Dashboard

* Accuracy comparison
* Precision / Recall / F1 comparison
* Training time analysis
* Experimental benchmarking charts

---

## 🏆 Conclusion Module

* Research findings
* Final verdict
* Comparative analysis
* Future scope

---

# 🖥️ Technology Stack

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | Core Development           |
| Streamlit    | Interactive Dashboard      |
| Scikit-learn | Machine Learning           |
| NLTK         | NLP Processing             |
| Gensim       | Word Embeddings            |
| Pandas       | Data Analysis              |
| NumPy        | Numerical Computing        |
| Plotly       | Interactive Visualizations |
| Matplotlib   | Graph Visualization        |

---

# 📁 Project Structure

```bash
NLP/
├── .gitignore
├── requirements.txt
├── README.md
├── .streamlit/
├── data/
│   ├── processed/
│   │   └── processed_legal_dataset_sample.csv
│   └── raw/
│       └── case_files_total.csv
├── embeddings/
│   ├── trained/
│   │   ├── doc2vec_dbow.model
│   │   ├── doc2vec_dm.model
│   │   ├── fasttext.model
│   │   ├── word2vec_cbow.model
│   │   └── word2vec_skipgram.model
│   └── vectorizers/
│       ├── bow_vectorizer.pkl
│       ├── onehot_vectorizer.pkl
│       └── tfidf_vectorizer.pkl
├── models/
│   └── trained/
│       └── [16 Trained Classifier PKLs]
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_analysis.ipynb
├── outputs/
│   ├── figures/
│   │   ├── comparison_accuracy.png
│   │   ├── comparison_f1_score.png
│   │   ├── confusion_matrices/
│   │   └── embedding_plots/
│   └── results/
│       └── model_comparison_results.csv
├── scripts/
│   └── run_model_comparison.py
├── src/
│   ├── embeddings/
│   ├── evaluation/
│   ├── experimental/
│   ├── models/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── generate_visuals.py
│       └── helpers.py
└── streamlit_app/
    └── app.py
```

---

# ▶️ Installation & Setup

## Step 1 — Clone Repository

```bash
git clone https://github.com/KoletiSankeerthana/NLP_Course_Project.git
```

---

## Step 2 — Navigate to Project Folder

```bash
cd NLP
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Run Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

# 📊 Experimental Outputs

### Generated Outputs Include

* Accuracy comparison plots
* F1-score comparison plots
* Confusion matrices
* Embedding visualizations
* Model benchmarking reports

Outputs are stored in:

```bash
outputs/
```

---

# 📘 Jupyter Notebooks

| Notebook                     | Purpose                     |
| ---------------------------- | --------------------------- |
| 01_data_preprocessing.ipynb  | Data preprocessing analysis |
| 02_embedding_analysis.ipynb  | Embedding comparison        |
| 03_model_training.ipynb      | Model training              |
| 04_evaluation_analysis.ipynb | Performance evaluation      |

---

# 🚀 Future Scope

The project can be extended using:

* Transformer Models (BERT / Legal-BERT)
* Explainable AI
* Semantic Legal Retrieval
* AI Legal Assistants
* Judgment Recommendation Systems
* Multilingual Legal NLP
* Legal Question Answering Systems

---

# ✅ Final Conclusion

This research demonstrates that sparse embedding techniques remain highly effective for structured legal document classification tasks.

However, dense embeddings provide superior semantic understanding and contextual representation for advanced Legal NLP applications.

### Final Experimental Verdict

| Category              | Best Model                     |
| --------------------- | ------------------------------ |
| Best Overall Accuracy | One-Hot + Random Forest        |
| Best Sparse Embedding | TF-IDF                         |
| Best Dense Embedding  | FastText                       |
| Best Dense Model      | FastText + Logistic Regression |

---

# 👨‍💻 Developed By

### Koleti Sankeerthana

B.Tech — School of Computing and Data Science
Sai University, Chennai

---

