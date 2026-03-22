import os
import sys

# Crucial bounds for TF stability on various machines
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, session
from config import Config
from database import init_db
from routes.auth_routes import auth_bp
from routes.prediction_routes import prediction_bp, init_model as init_pred_model
from routes.dataset_routes import dataset_bp, init_model as init_dataset_model
from routes.analytics_routes import analytics_bp, init_model as init_analytics_model
from models.multilingual_model import MultilingualModel
from utils.limiter import limiter


def create_app():
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')

    app.config.from_object(Config)

    # Initialize database
    init_db()

    # Initialize rate limiter
    limiter.init_app(app)

    # Create necessary directories
    os.makedirs(Config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    # Initialize ML model
    model = MultilingualModel()
    if not model.load():
        print("[WARN] Failed to load Multilingual Model.")

    # Inject model into route modules
    init_pred_model(model)
    init_dataset_model(model)
    init_analytics_model(model)

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(analytics_bp)

    # Home route
    @app.route('/')
    def index():
        return render_template('index.html')

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return render_template('index.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('index.html'), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
