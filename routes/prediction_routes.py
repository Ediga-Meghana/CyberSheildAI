import logging
from flask import Blueprint, request, jsonify, session, render_template, flash, redirect, url_for
from database import insert_db, query_db
from utils.language_detector import detect_language, SUPPORTED_LANGUAGES
from utils.translator import translate_to_english
from utils.limiter import limiter

prediction_bp = Blueprint('prediction', __name__)

# Configure logging for predictions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
@limiter.limit("10 per minute")
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized. Please log in.'}), 401

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided.'}), 400

    text = data['text'].strip()
    force_lang = data.get('language', 'auto')
    
    # Input validation and sanitization
    if not text or len(text) > 3000:
        return jsonify({'error': 'Invalid text length. Must be 1 to 3000 characters.'}), 400

    # Detect language
    if force_lang and force_lang != 'auto' and force_lang in SUPPORTED_LANGUAGES:
        lang_code = force_lang
        lang_name = SUPPORTED_LANGUAGES[lang_code]
    else:
        lang_code, lang_name = detect_language(text)

    # Translate if not English
    processed_text = text
    if lang_code != 'en':
        processed_text = translate_to_english(text, source_lang=lang_code)

    # Predict
    try:
        result = model.predict(processed_text)
        
        # Explainability (Bonus)
        if getattr(model, 'is_trained', False) and getattr(model, 'model', None) is not None:
            # Mock explanation for UI Preview without LIME
            words = processed_text.split()
            mock_exp = []
            for w in words[:5]:
                # Assign dummy positive weight if prediction is bullying
                mock_exp.append((w, 0.15 if result['label'] == 1 else -0.15))
            result['explanation'] = mock_exp
        else:
            result['explanation'] = []
            
    except Exception as e:
        logger.error(f"Prediction failed for user {session['user_id']}: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Logging predictions for future retraining (Bonus)
    logger.info(f"PREDICTION_LOG | user_id:{session['user_id']} | text:'{text}' | pred:{result['prediction']} | conf:{result['confidence']} | lang:{lang_code}")

    # Save to database
    # Assuming the table structure remains mostly same. 
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
