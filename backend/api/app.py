"""
app.py — Flask REST API for Scent Duration Prediction
Serves predictions, ingredient recommendations, and model info.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_service import ModelService

app = Flask(__name__)
CORS(app)

# Initialize model service
service = ModelService()


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_ready': service.is_ready()})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict scent duration from fragrance composition."""
    if not service.is_ready():
        return jsonify({'error': 'Model not loaded. Run main.py first.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    try:
        result = service.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ingredients', methods=['GET'])
def get_ingredients():
    """Get available ingredients grouped by note type."""
    return jsonify(service.get_ingredients())


@app.route('/api/options', methods=['GET'])
def get_options():
    """Get fragrance families and application types."""
    return jsonify(service.get_options())


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model performance metrics and comparison."""
    return jsonify(service.get_model_info())


@app.route('/api/optimize', methods=['GET'])
def get_optimal_combinations():
    """Get optimal ingredient combinations for maximum longevity."""
    limit = request.args.get('limit', 15, type=int)
    combos = service.get_optimal_combinations(limit)
    contributions = service.get_ingredient_contributions()
    return jsonify({
        'optimal_combinations': combos,
        'ingredient_contributions': contributions,
    })


@app.route('/api/plots/<filename>', methods=['GET'])
def get_plot(filename):
    """Serve global explainability plots."""
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    if not os.path.exists(os.path.join(plots_dir, filename)):
        return jsonify({'error': 'Plot not found'}), 404
    return send_from_directory(plots_dir, filename)


if __name__ == '__main__':
    print("=" * 50)
    print("  Scent Duration Prediction API")
    print(f"  Model ready: {service.is_ready()}")
    print("=" * 50)
    app.run(debug=True, port=5000, host='0.0.0.0')
