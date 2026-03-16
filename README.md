# CyberShield AI — Cyberbullying Detection System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Flask-3.0.0-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/Framework-Hybrid%20LSTM%2BSVM-orange.svg" alt="ML Framework">
  <img src="https://img.shields.io/badge/UI-Dark%20Glassmorphism-magenta.svg" alt="UI Theme">
</div>

<br>

CyberShield AI is a state-of-the-art, production-ready web application that detects cyberbullying in social media text across multiple languages. It implements an advanced research-based architecture combining deep learning and classical machine learning to provide accurate, real-time predictions.

## 🚀 Features

- **Hybrid ML Architecture**: Combines Bi-directional LSTM for deep contextual feature extraction and SVM for robust classification.
- **Multilingual Support**: Automatically detects and analyzes text in English, Hindi, Telugu, Spanish, and more.
- **Synthetic Data Generation**: Overcomes dataset imbalance (a common IEEE research limitation) using SMOTE, NLP augmentation, and template-based text generation to improve minority class recall.
- **Real-Time Detection**: Paste or type any text to instantly see the prediction, sub-category, and confidence score.
- **Analytics Dashboard**: Interactive Chart.js visualizations covering model accuracy, recall, F1-scores, confusion matrix, and prediction history.
- **Modern UI**: A sleek, dark-themed glassmorphism interface with smooth animations and responsive design.
- **Dataset Management**: Upload custom CSV datasets and trigger on-demand model training.

## 📌 Addressed Research Limitations
This project specifically addresses common drawbacks found in IEEE research papers regarding cyberbullying detection:
1. **Minority Class Recall**: Solved via 50/50 balanced synthetic data generation and targeted minority-class augmentation (synonym replacement, etc.).
2. **Computational Efficiency**: Optimizes training using a lightweight extraction pipeline and early stopping, allowing the model to train in seconds even on CPU.
3. **Multilingual Applicability**: Integrates `langdetect` and `deep-translator` to expand detection capabilities beyond English.

## 🛠️ Technology Stack
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript, Chart.js
- **Backend & API**: Python, Flask, SQLite
- **Machine Learning**: TensorFlow/Keras (BiLSTM), Scikit-Learn (SVM, Logistic Regression), NLTK, Imbalanced-Learn
- **NLP Utilities**: TF-IDF (Character N-grams), deep-translator, langdetect

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ediga-Meghana/cyberAI.git
   cd cyberAI
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: NLTK dependencies are handled internally through built-in alternatives to prevent Windows download issues.*

3. **Run the Flask server:**
   ```bash
   python app.py
   ```
   *Note: The first time you run this, the system will automatically generate a balanced synthetic dataset and train the hybrid model.*

4. **Access the application:**
   Open your browser and navigate to: `http://127.0.0.1:5000`

## 🧪 Testing

You can use the built-in system without an external dataset.
- Register a new account or use the test credentials if you set them up.
- Go to the **Detect** page and try pasting common examples of safe text or bullying text (e.g., Harassment, Hate Speech, Threats).

## 🗂️ Project Structure
- `/models`: Contains the Hybrid LSTM+SVM logic.
- `/preprocessing`: Text cleaning, tokenization, and character-level TF-IDF extraction.
- `/routes`: Flask blueprint routes (auth, prediction, analytics, dataset).
- `/synthetic`: Custom data generator and NLP augmentation engines.
- `/static/css`: Custom UI styles (`style.css`).
- `/templates`: HTML pages.

## 📄 License
This project is for educational and research demonstration purposes.
