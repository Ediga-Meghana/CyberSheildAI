import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cyberbully-detect-secret-key-2024')
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'cyberbully.db')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    TFIDF_MAX_FEATURES = 10000
    TFIDF_NGRAM_RANGE = (2, 5)
    LSTM_EMBEDDING_DIM = 128
    LSTM_VOCAB_SIZE = 20000
    LSTM_MAX_LEN = 200
    LSTM_UNITS = 64
    BATCH_SIZE = 32
    EPOCHS = 20
    LABELS = ['not_cyberbullying', 'cyberbullying']
    CATEGORIES = ['Not Bullying', 'Harassment', 'Hate Speech', 'Threat', 'Identity Attack']
