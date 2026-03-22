import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModel
from config import Config
from preprocessing.clean_text import clean_text

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3

def train_pipeline(data_path="datasets/real_dataset.csv"):
    os.makedirs(Config.SAVED_MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset {data_path} not found. Using a dummy sample to demonstrate pipeline execution.")
        # Create a tiny dummy dataset to prevent crashing, demonstrating capability
        df = pd.DataFrame({
            'text': ['I love you', 'You are stupid and I will kill you', 'Hello world', 'Go die in a hole', 'Nice weather'] * 20,
            'label': [0, 1, 0, 1, 0] * 20
        })
    else:
        df = pd.read_csv(data_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    print(f"[INFO] Loaded {len(df)} samples.")
    df['clean_text'] = df['text'].apply(lambda x: clean_text(str(x)))
    
    X = df['clean_text'].tolist()
    y = df['label'].tolist()
    
    # REQUIREMENT: Proper train/validation/test split (70/15/15)
    # 1. Split 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # 2. Split Temp into 15% Val, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    print(f"[INFO] Data Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def encode_data(texts):
        return tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='np'
        )

    # Convert to NumPy dicts
    train_encoded = encode_data(X_train)
    val_encoded = encode_data(X_val)
    test_encoded = encode_data(X_test)

    # ---------------------------------------------------------
    # PART 1: Logistic Regression with GridSearchCV & L1/L2
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 1: Extracting BERT Embeddings & Tuning Logistic Regression")
    print("="*50)
    
    base_model = TFAutoModel.from_pretrained(MODEL_NAME)
    
    def get_embeddings(encoded_dict):
        # We process in batches to avoid OOM
        embeds = []
        inputs = encoded_dict['input_ids']
        masks = encoded_dict['attention_mask']
        for i in range(0, len(inputs), BATCH_SIZE):
            batch_inputs = inputs[i:i+BATCH_SIZE]
            batch_masks = masks[i:i+BATCH_SIZE]
            out = base_model(input_ids=batch_inputs, attention_mask=batch_masks)
            # Use [CLS] token
            embeds.append(out.last_hidden_state[:, 0, :].numpy())
        return np.vstack(embeds)
    
    print("[INFO] Extracting embeddings for train...")
    X_train_emb = get_embeddings(train_encoded)
    print("[INFO] Extracting embeddings for test...")
    X_test_emb = get_embeddings(test_encoded)
    
    # Hyperparameter tuning with GridSearchCV & 5-fold CV
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'] # Liblinear supports l1
    }
    
    print("[INFO] Running GridSearchCV with 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    gs = GridSearchCV(lr, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    gs.fit(X_train_emb, y_train)
    
    best_lr = gs.best_estimator_
    print(f"[INFO] Best Logistic Regression Model params: {gs.best_params_}")
    
    # Save best LR model
    joblib.dump(best_lr, os.path.join(Config.SAVED_MODELS_DIR, 'best_lr_model.pkl'))
    
    # LR Predictions & Metrics
    lr_pred = best_lr.predict(X_test_emb)
    lr_prob = best_lr.predict_proba(X_test_emb)[:, 1]
    
    print("\n--- Logistic Regression Metrics ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
    print("ROC-AUC:", roc_auc_score(y_test, lr_prob))
    print("Classification Report:\n", classification_report(y_test, lr_pred, zero_division=0))
    

    # ---------------------------------------------------------
    # PART 2: Fine-Tuning Transformer Model
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 2: Fine-Tuning Transformer for Classification")
    print("="*50)
    
    # Delete base_model to free RAM
    del base_model
    tf.keras.backend.clear_session()
    
    # Prepare TF Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encoded), y_train)).shuffle(len(y_train)).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encoded), y_val)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encoded), y_test)).batch(BATCH_SIZE)
    
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Train
    print("[INFO] Fine-tuning the Transformer model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )
    
    # Evaluate
    print("[INFO] Evaluating Fine-tuned model on test set...")
    predictions = model.predict(test_dataset).logits
    tf_probs = tf.nn.softmax(predictions, axis=1).numpy()
    tf_preds = np.argmax(tf_probs, axis=1)
    
    conf_matrix = confusion_matrix(y_test, tf_preds)
    roc_auc = roc_auc_score(y_test, tf_probs[:, 1])
    report = classification_report(y_test, tf_preds, output_dict=True, zero_division=0)
    
    print("\n--- Fine-Tuned Transformer Metrics ---")
    print("Confusion Matrix:\n", conf_matrix)
    print("ROC-AUC:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, tf_preds, zero_division=0))
    
    metrics_dict = {
        'accuracy': float(report['accuracy']),
        'roc_auc': float(roc_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report
    }
    
    # Save transformer
    save_path = os.path.join(Config.SAVED_MODELS_DIR, 'transformer_finetuned')
    model.save_pretrained(save_path)
    
    with open(os.path.join(Config.SAVED_MODELS_DIR, 'transformer_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print(f"\n[INFO] Models and metrics saved successfully to {Config.SAVED_MODELS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/real_dataset.csv", help="Path to CSV dataset")
    args = parser.parse_args()
    
    train_pipeline(args.data)
