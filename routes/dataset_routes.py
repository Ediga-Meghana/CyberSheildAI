import os
import pandas as pd
from flask import Blueprint, request, jsonify, session, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from database import insert_db
from config import Config

dataset_bp = Blueprint('dataset', __name__)

# Model reference injected from app.py
model = None


def init_model(m):
    global model
    model = m


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@dataset_bp.route('/dataset', methods=['GET'])
def dataset_page():
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('auth.login'))
    return render_template('dataset.html')


@dataset_bp.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only CSV and TXT allowed.'}), 400

    filename = secure_filename(file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Get file info
    try:
        df = pd.read_csv(filepath)
        num_rows = len(df)
    except Exception:
        num_rows = 0

    file_size = os.path.getsize(filepath)

    insert_db(
        'INSERT INTO datasets (dataset_name, file_path, size, num_rows, uploaded_by) VALUES (?, ?, ?, ?, ?)',
        (filename, filepath, file_size, num_rows, session['user_id'])
    )

    return jsonify({
        'message': 'Dataset uploaded successfully!',
        'filename': filename,
        'size': file_size,
        'rows': num_rows
    })


@dataset_bp.route('/train_model', methods=['POST'])
def train_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if session.get('role') != 'admin' and session.get('user_id') != 1:
        # Allow first user (admin) and users with admin role
        pass  # For demo, allow anyone to train

    try:
        # Check if a dataset was provided
        data = request.get_json() or {}
        dataset_name = data.get('dataset_name', None)

        if dataset_name:
            filepath = os.path.join(Config.UPLOAD_FOLDER, dataset_name)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                texts = df['text'].tolist() if 'text' in df.columns else df.iloc[:, 0].tolist()
                labels = df['label'].tolist() if 'label' in df.columns else df.iloc[:, 1].tolist()
                categories = df['category'].tolist() if 'category' in df.columns else [None] * len(texts)
                metrics = model.train(texts, labels, categories)
            else:
                return jsonify({'error': 'Dataset not found.'}), 404
        else:
            # Train on synthetic data
            metrics = model.train()

        return jsonify({
            'message': 'Model trained successfully!',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500
