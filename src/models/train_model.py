"""
Model Training Module
---------------------
Provides a standardized interface for training various classification models
across different embedding types.

Author: Antigravity (AI Assistant)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time
import os
import joblib
from src.utils.config import TRAINED_MODELS_PATH

class ModelTrainer:
    """
    Handles training and serialization of machine learning models.
    """
    def __init__(self, model_type='logistic_regression', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == 'svm':
            return SVC(kernel='linear', probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train, y_train):
        """
        Trains the model and returns the training time.
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        return end_time - start_time

    def save_model(self, embedding_name):
        """
        Saves the model using a professional naming convention.
        Format: EmbeddingName_ModelType.pkl
        """
        if not os.path.exists(TRAINED_MODELS_PATH):
            os.makedirs(TRAINED_MODELS_PATH)
            
        filename = f"{embedding_name}_{self.model_type}.pkl"
        path = os.path.join(TRAINED_MODELS_PATH, filename)
        joblib.dump(self.model, path)
        return path

if __name__ == "__main__":
    # Test
    trainer = ModelTrainer('logistic_regression')
    print(f"Model Initialized: {trainer.model}")
