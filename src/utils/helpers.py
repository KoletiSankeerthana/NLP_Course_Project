"""
Utility Module
--------------
Shared helper functions for logging, file management, and configuration.

Author: Antigravity (AI Assistant)
"""

import logging
import os
import nltk

def setup_logger(name: str, level=logging.INFO):
    """
    Configures a professional logger for the project.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger

def ensure_dir(path: str):
    """
    Ensures that a directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_nltk():
    """
    Safely checks and downloads required NLTK resources.
    Avoids repeated downloads and handles missing resources gracefully.
    """
    resources = {
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4'
    }
    
    print("--- Initializing NLTK Resources ---")
    for path, name in resources.items():
        try:
            nltk.data.find(path)
            # print(f"Resource '{name}' found.")
        except LookupError:
            print(f"Resource '{name}' not found. Downloading...")
            nltk.download(name, quiet=True)
    print("NLTK Initialization Complete.")
