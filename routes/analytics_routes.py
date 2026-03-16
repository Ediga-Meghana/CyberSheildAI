from flask import Blueprint, jsonify, session, render_template, redirect, url_for, flash
from database import query_db

analytics_bp = Blueprint('analytics', __name__)

# Model reference injected from app.py
model = None


def init_model(m):
    global model
    model = m


@analytics_bp.route('/dashboard', methods=['GET'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('auth.login'))
    return render_template('dashboard.html')


@analytics_bp.route('/analytics', methods=['GET'])
def get_analytics():
    """Return model metrics and prediction statistics."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    # Model metrics
    metrics = model.metrics if model and model.is_trained else {}

    # Prediction stats
    total_predictions = query_db('SELECT COUNT(*) as count FROM predictions', one=True)
    bullying_count = query_db("SELECT COUNT(*) as count FROM predictions WHERE prediction = 'Cyberbullying'", one=True)
    not_bullying_count = query_db("SELECT COUNT(*) as count FROM predictions WHERE prediction = 'Not Cyberbullying'", one=True)

    # Category distribution
    category_dist = query_db(
        'SELECT category, COUNT(*) as count FROM predictions GROUP BY category ORDER BY count DESC'
    )

    # Recent predictions
    recent = query_db(
        'SELECT input_text, prediction, category, confidence, language, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 10'
    )

    # Language distribution
    lang_dist = query_db(
        'SELECT language, COUNT(*) as count FROM predictions GROUP BY language ORDER BY count DESC'
    )

    return jsonify({
        'model_metrics': metrics,
        'total_predictions': total_predictions['count'] if total_predictions else 0,
        'bullying_count': bullying_count['count'] if bullying_count else 0,
        'not_bullying_count': not_bullying_count['count'] if not_bullying_count else 0,
        'category_distribution': [dict(r) for r in category_dist] if category_dist else [],
        'recent_predictions': [dict(r) for r in recent] if recent else [],
        'language_distribution': [dict(r) for r in lang_dist] if lang_dist else []
    })


@analytics_bp.route('/admin', methods=['GET'])
def admin():
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('auth.login'))
    return render_template('admin.html')


@analytics_bp.route('/admin/data', methods=['GET'])
def admin_data():
    """Return admin panel data."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    users = query_db('SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC')
    predictions = query_db(
        '''SELECT p.id, u.username, p.input_text, p.prediction, p.category, p.confidence, p.timestamp
           FROM predictions p LEFT JOIN users u ON p.user_id = u.id
           ORDER BY p.timestamp DESC LIMIT 100'''
    )
    datasets = query_db(
        '''SELECT d.id, d.dataset_name, d.size, d.num_rows, u.username, d.upload_date
           FROM datasets d LEFT JOIN users u ON d.uploaded_by = u.id
           ORDER BY d.upload_date DESC'''
    )

    return jsonify({
        'users': [dict(r) for r in users] if users else [],
        'predictions': [dict(r) for r in predictions] if predictions else [],
        'datasets': [dict(r) for r in datasets] if datasets else []
    })
