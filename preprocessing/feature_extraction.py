import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config


class FeatureExtractor:
    """Character-level TF-IDF feature extractor."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            max_features=Config.TFIDF_MAX_FEATURES,
            sublinear_tf=True
        )
        self.is_fitted = False

    def fit(self, texts):
        """Fit the TF-IDF vectorizer on texts."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        """Transform texts to TF-IDF features."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """Fit and transform texts."""
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts)

    def save(self, path=None):
        """Save vectorizer to disk."""
        if path is None:
            os.makedirs(Config.SAVED_MODELS_DIR, exist_ok=True)
            path = os.path.join(Config.SAVED_MODELS_DIR, 'tfidf_vectorizer.pkl')
        joblib.dump(self.vectorizer, path)

    def load(self, path=None):
        """Load vectorizer from disk."""
        if path is None:
            path = os.path.join(Config.SAVED_MODELS_DIR, 'tfidf_vectorizer.pkl')
        if os.path.exists(path):
            self.vectorizer = joblib.load(path)
            self.is_fitted = True
            return True
        return False
