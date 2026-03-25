import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from synthetic.synthetic_generator import SyntheticDataGenerator
from preprocessing.clean_text import clean_text

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
MAX_VOCAB = 5000
MAX_LEN = 100
EMBEDDING_DIM = 64
EPOCHS = 5
BATCH_SIZE = 32

def run_evaluation():
    print("=========================================")
    print("1. DATA PREPARATION")
    print("=========================================")
    # Generate balanced synthetic data
    print("Generating dataset...")
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_dataset(total_size=3000)
    
    # Preprocess
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(lambda x: clean_text(str(x)))
    X_text = df['clean_text'].values
    y = df['label'].values

    # Train/Test Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class Weights applied for imbalance: {class_weight_dict}")

    print("\n=========================================")
    print("2. FEATURE EXTRACTION")
    print("=========================================")
    
    # A. TF-IDF (Character-level) for SVM
    print("Extracting Character-level TF-IDF features for SVM...")
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    # B. Tokenizer for LSTM
    print("Filtering vocabulary and tokenizing for LSTM...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post', truncating='post')

    print("\n=========================================")
    print("3. MODEL TRAINING (ABLATION STUDY)")
    print("=========================================")

    # Model 1: Support Vector Machine (SVM)
    print("Training SVM (Model 1)...")
    svm_model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    
    # Model 2: LSTM
    print("Training LSTM (Model 2)...")
    lstm_model = Sequential([
        Embedding(input_dim=MAX_VOCAB, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5), # Regularization to prevent overfitting
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    lstm_model.fit(
        X_train_seq, y_train, 
        epochs=EPOCHS, batch_size=BATCH_SIZE, 
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stop], 
        verbose=1
    )

    print("\n=========================================")
    print("4. EVALUATION & HYBRID ENSEMBLE")
    print("=========================================")
    
    # Predictions - SVM
    svm_probs = svm_model.predict_proba(X_test_tfidf)[:, 1]
    svm_preds = (svm_probs >= 0.5).astype(int)

    # Predictions - LSTM
    lstm_probs = lstm_model.predict(X_test_seq).flatten()
    lstm_preds = (lstm_probs >= 0.5).astype(int)

    # Predictions - Hybrid (Weighted Average: 0.6 LSTM + 0.4 SVM)
    hybrid_probs = (lstm_probs * 0.6) + (svm_probs * 0.4)
    hybrid_preds = (hybrid_probs >= 0.5).astype(int)

    def calculate_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        }

    metrics_svm = calculate_metrics(y_test, svm_preds)
    metrics_lstm = calculate_metrics(y_test, lstm_preds)
    metrics_hybrid = calculate_metrics(y_test, hybrid_preds)

    # Print Ablation Table
    print("\n--- ABLATION STUDY RESULTS ---")
    df_metrics = pd.DataFrame([metrics_svm, metrics_lstm, metrics_hybrid], index=["SVM Only", "LSTM Only", "Hybrid (LSTM+SVM)"])
    print(df_metrics.to_markdown())

    df_metrics.to_csv(f"{OUTPUT_DIR}/ablation_metrics.csv")

    # Save Confusion Matrices
    def plot_cm(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Not Bullying (0)', 'Bullying (1)'], 
                    yticklabels=['Not Bullying (0)', 'Bullying (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{filename}.png")
        plt.close()

    plot_cm(y_test, svm_preds, "Confusion Matrix: SVM", "cm_svm")
    plot_cm(y_test, lstm_preds, "Confusion Matrix: LSTM", "cm_lstm")
    plot_cm(y_test, hybrid_preds, "Confusion Matrix: Hybrid LSTM+SVM", "cm_hybrid")

    print("\n=========================================")
    print("5. ERROR ANALYSIS")
    print("=========================================")
    # Identify False Positives and False Negatives from Hybrid
    fn_indices = np.where((y_test == 1) & (hybrid_preds == 0))[0]
    fp_indices = np.where((y_test == 0) & (hybrid_preds == 1))[0]

    print(f"\nTotal False Negatives (Missed Bullying): {len(fn_indices)}")
    for i in fn_indices[:3]:
        print(f"  - True: 1, Pred: 0 | Prob: {hybrid_probs[i]:.2f} -> Text: {X_test_text[i]}")

    print(f"\nTotal False Positives (False Alarms): {len(fp_indices)}")
    for i in fp_indices[:3]:
        print(f"  - True: 0, Pred: 1 | Prob: {hybrid_probs[i]:.2f} -> Text: {X_test_text[i]}")

    print("\n=========================================")
    print("6. ROBUSTNESS & REAL-WORLD TESTING")
    print("=========================================")
    test_cases = [
        "I absolutely love you, you're the best!",  # Clear safe
        "Go kill yourself, you worthless trash.",  # Clear explicit bullying
        "Wow, you're a genius.",                   # Sarcasm (Hard) - Safe tone but could be mean
        "You look like a monkey, go back to the zoo.", # Implicit racism/bullying
        "I'm just kidding bro, ur kinda slow tho." # Slang / Mixed tone
    ]

    clean_test_cases = [clean_text(t) for t in test_cases]
    test_tfidf = tfidf.transform(clean_test_cases)
    test_seq = pad_sequences(tokenizer.texts_to_sequences(clean_test_cases), maxlen=MAX_LEN, padding='post', truncating='post')

    s_probs = svm_model.predict_proba(test_tfidf)[:, 1]
    l_probs = lstm_model.predict(test_seq).flatten()
    h_probs = (l_probs * 0.6) + (s_probs * 0.4)
    
    for i, text in enumerate(test_cases):
        pred_label = "Bullying" if h_probs[i] >= 0.5 else "Safe"
        print(f"\nSentence: '{text}'")
        print(f"  Hybrid Prob: {h_probs[i]:.4f} -> {pred_label}")
        print(f"  [LSTM Prob: {l_probs[i]:.4f} | SVM Prob: {s_probs[i]:.4f}]")

    print("\n[SUCCESS] Evaluation Complete. Models and artifacts saved.")

if __name__ == '__main__':
    run_evaluation()
