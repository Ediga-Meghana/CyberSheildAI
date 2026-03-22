import os
import json
import random
from config import Config
from preprocessing.clean_text import clean_text

class AdvancedModel:
    """Mock model to allow UI testing without crashing on TensorFlow imports."""
    
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.model_name = model_name
        self.is_trained = True
        self.model = True # Mock flag to enable explainability UI
        self.metrics = {}
        self.category_map = {i: cat for i, cat in enumerate(Config.CATEGORIES)}
        
    def encode_texts(self, texts, max_length=128):
        return texts
        
    def train(self, texts=None, labels=None, categories=None):
        pass

    def load(self):
        self.is_trained = True
        return True

    def predict(self, text):
        cleaned = clean_text(text)
        text_lower = text.lower()
        
        # Simple heuristic to simulate ML decisions for preview
        is_bullying = any(word in text_lower for word in ['stupid', 'idiot', 'kill', 'hate', 'loser', 'die'])
        pred_label = 1 if is_bullying else 0
        confidence = 0.82 + (random.random() * 0.17)
        
        category = "Not Bullying"
        if pred_label == 1:
            if 'kill' in text_lower or 'die' in text_lower:
                category = 'Threat'
            elif 'hate' in text_lower:
                category = 'Hate Speech'
            else:
                category = 'Harassment'
        
        return {
            'prediction': 'Cyberbullying' if pred_label == 1 else 'Not Cyberbullying',
            'label': pred_label,
            'confidence': round(confidence, 4),
            'category': category
        }
        
    def predict_batch(self, texts):
        return [self.predict(t) for t in texts]
