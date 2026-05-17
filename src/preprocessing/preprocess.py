"""
Legal Text Preprocessing Pipeline
---------------------------------
Research-quality NLP preprocessing pipeline for legal document classification.
"""

import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser


# =========================================================
# NLTK RESOURCE DOWNLOAD
# =========================================================

def ensure_nltk_resources():
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]

    for resource in resources:
        try:
            nltk.data.find(resource)
        except:
            nltk.download(resource, quiet=True)


ensure_nltk_resources()


# =========================================================
# GLOBAL OBJECTS
# =========================================================

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text: str) -> str:
    """
    Professional legal text cleaning.
    Removes OCR artifacts and noisy metadata.
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove html
    text = re.sub(r'<[^>]+>', ' ', text)

    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # =====================================================
    # REMOVE OCR GARBAGE
    # =====================================================

    # remove patterns like:
    # .,2398.0,74.0,ACC
    # 12.0,44.0,A
    # .,.,.,XYZ

    text = re.sub(
        r'([.,\s]*\d+\.\d+[.,]*)+[a-zA-Z]*',
        ' ',
        text
    )

    # remove isolated decimals
    text = re.sub(r'\b\d+\.\d+\b', ' ', text)

    # remove long numeric chains
    text = re.sub(r'\b\d+(?:[.,]\d+){1,}\b', ' ', text)

    # remove OCR punctuation chains
    text = re.sub(r'[.,]{2,}', ' ', text)

    # remove OCR numeric-word chains like:
    # ,2398.0,74.0,Accepted
    text = re.sub(
        r'[",.\s]*\d+\.\d+(?:,\d+\.\d+)*(?:,[a-zA-Z]+)+',
        ' ',
    text
)

    # remove weird OCR mixed fragments
    text = re.sub(
        r'\b[a-zA-Z]*\d+[a-zA-Z]*\d+[a-zA-Z]*\b',
        ' ',
        text
    )

    # =====================================================
    # PRESERVE LEGAL REFERENCES
    # =====================================================

    # remove very long meaningless numbers
    text = re.sub(r'\b\d{5,}\b', ' ', text)

    # =====================================================
    # REMOVE PUNCTUATION
    # =====================================================

    text = re.sub(r'[^\w\s]', ' ', text)

    # remove isolated single chars
    text = re.sub(r'\b[b-hj-z]\b', ' ', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize_text(text: str) -> list:

    if not isinstance(text, str):
        return []

    return word_tokenize(text)


# =========================================================
# FILTER TOKENS
# =========================================================

def filter_tokens(tokens: list) -> list:
    """
    Removes OCR garbage and noisy tokens.
    """

    filtered = []

    legal_keep = {
        "ipc",
        "crpc",
        "cpc",
        "fir",
        "section",
        "article",
        "act"
    }

    for token in tokens:

        token = token.strip().lower()

        if not token:
            continue

        # preserve legal words
        if token in legal_keep:
            filtered.append(token)
            continue

        # remove tokens without letters
        if not re.search(r'[a-z]', token):
            continue

        # remove floating number fragments
        if re.search(r'\d+\.\d+', token):
            continue

        # remove OCR mixed garbage
        if re.search(r'[a-z]*\d+[a-z]*\d+', token):
            continue

        # remove punctuation-only tokens
        if re.fullmatch(r'[\W\d_]+', token):
            continue

        # remove very short useless tokens
        if len(token) == 1 and token not in ['a', 'i']:
            continue

        # remove extremely long garbage
        if len(token) > 25:
            continue

        filtered.append(token)

    return filtered


# =========================================================
# STOPWORD REMOVAL
# =========================================================

def remove_stopwords(tokens: list) -> list:

    stop_words = set(stopwords.words('english'))

    preserve_words = {
        'against',
        'before',
        'under',
        'between'
    }

    stop_words = stop_words - preserve_words

    return [
        word for word in tokens
        if word not in stop_words
    ]


# =========================================================
# STEMMING
# =========================================================

def stem_text(tokens: list) -> list:

    return [stemmer.stem(word) for word in tokens]


# =========================================================
# LEMMATIZATION
# =========================================================

def lemmatize_text(tokens: list) -> list:

    return [lemmatizer.lemmatize(word) for word in tokens]


# =========================================================
# POS TAGGING
# =========================================================

def pos_tag_text(tokens: list):

    if not tokens:
        return []

    return nltk.pos_tag(tokens)


# =========================================================
# PARSING
# =========================================================

def parse_text(tagged_tokens):

    if not tagged_tokens:
        return None

    grammar = r"""
        NP: {<DT|JJ|NN.*>+}
        VP: {<VB.*><NP|PP|CLAUSE>+$}
    """

    parser = RegexpParser(grammar)

    return parser.parse(tagged_tokens)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def preprocess_pipeline(text: str) -> str:

    # clean
    cleaned = clean_text(text)

    # normalize
    normalized = normalize_text(cleaned)

    # tokenize
    tokens = tokenize_text(normalized)

    # filter tokens
    tokens = filter_tokens(tokens)

    # remove stopwords
    tokens = remove_stopwords(tokens)

    # lemmatize
    tokens = lemmatize_text(tokens)

    # final cleanup
    tokens = [
        token for token in tokens
        if len(token) > 1
    ]

    return " ".join(tokens)


# =========================================================
# DATAFRAME PROCESSING
# =========================================================

def process_dataframe(
    df: pd.DataFrame,
    text_column: str,
    output_path: str = None
):

    if text_column not in df.columns:
        raise ValueError(f"{text_column} not found.")

    df[text_column] = df[text_column].fillna("")

    print("Applying preprocessing pipeline...")

    df['processed_text'] = df[text_column].apply(
        lambda x: preprocess_pipeline(str(x))
    )

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return df


# =========================================================
# DEMONSTRATION
# =========================================================

def demonstrate_pipeline(sample_text: str):

    print("=" * 60)

    print("\nRAW:")
    print(sample_text)

    cleaned = clean_text(sample_text)

    print("\nCLEANED:")
    print(cleaned)

    normalized = normalize_text(cleaned)

    print("\nNORMALIZED:")
    print(normalized)

    tokens = tokenize_text(normalized)

    print("\nTOKENS:")
    print(tokens)

    filtered = filter_tokens(tokens)

    print("\nFILTERED:")
    print(filtered)

    stop_removed = remove_stopwords(filtered)

    print("\nSTOPWORDS REMOVED:")
    print(stop_removed)

    stemmed = stem_text(stop_removed)

    print("\nSTEMMED:")
    print(stemmed)

    lemmatized = lemmatize_text(stop_removed)

    print("\nLEMMATIZED:")
    print(lemmatized)

    pos_tags = pos_tag_text(lemmatized)

    print("\nPOS TAGS:")
    print(pos_tags)

    parsed = parse_text(pos_tags)

    print("\nPARSED:")
    print(parsed)

    final_output = preprocess_pipeline(sample_text)

    print("\nFINAL OUTPUT:")
    print(final_output)

    print("=" * 60)
# =========================================================
# WRAPPER CLASS FOR DEPLOYMENT/INTERFACES
# =========================================================

class TextPreprocessor:
    def __init__(self):
        ensure_nltk_resources()
        
    def clean_text(self, text: str) -> str:
        return clean_text(text)
        
    def tokenize(self, text: str) -> list:
        return tokenize_text(text)
        
    def remove_stopwords(self, tokens: list) -> list:
        return remove_stopwords(tokens)
        
    def lemmatize(self, tokens: list) -> list:
        return lemmatize_text(tokens)
        
    def full_preprocess(self, text: str) -> str:
        return preprocess_pipeline(text)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    sample_text = """
    .,2398.0,74.0,ACC Criminal Appeal was filed
    in the High Court under section 302 IPC.
    """

    demonstrate_pipeline(sample_text)

    try:

        df = pd.read_csv(
            "data/raw/case_files_total.csv"
        )

        possible_columns = [
            'judgement',
            'case_info',
            'text',
            'content'
        ]

        target_column = None

        for col in possible_columns:
            if col in df.columns:
                target_column = col
                break

        if target_column is None:
            target_column = df.columns[0]

        process_dataframe(
            df,
            target_column,
            output_path="data/processed/processed_legal_dataset.csv"
        )

        print("\nDataset preprocessing completed successfully.")

    except Exception as e:
        print(f"\nERROR: {e}")