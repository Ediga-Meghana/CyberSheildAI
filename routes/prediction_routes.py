from flask import Blueprint, request, jsonify, session, render_template, flash, redirect, url_for
from database import insert_db, query_db
from utils.language_detector import detect_language
from utils.translator import translate_to_english

prediction_bp = Blueprint('prediction', __name__)

# The model will be injected from app.py
model = None


def init_model(m):
    global model
    model = m


@prediction_bp.route('/detect', methods=['GET'])
def detect():
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('auth.login'))
    return render_template('detect.html')


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized. Please log in.'}), 401

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided.'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty.'}), 400

    # Detect language
    lang_code, lang_name = detect_language(text)

    # Translate if not English
    processed_text = text
    if lang_code != 'en':
        processed_text = translate_to_english(text, source_lang=lang_code)

    # Predict
    try:
        result = model.predict(processed_text)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Save to database
    insert_db(
        'INSERT INTO predictions (user_id, input_text, prediction, category, confidence, language) VALUES (?, ?, ?, ?, ?, ?)',
        (session['user_id'], text, result['prediction'], result['category'], result['confidence'], lang_code)
    )

    result['language'] = lang_name
    result['original_text'] = text
    if lang_code != 'en':
        result['translated_text'] = processed_text

    return jsonify(result)


@prediction_bp.route('/history', methods=['GET'])
def history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    records = query_db(
        'SELECT input_text, prediction, category, confidence, language, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50',
        (session['user_id'],)
    )
    history_list = [dict(r) for r in records]
    return jsonify(history_list)
