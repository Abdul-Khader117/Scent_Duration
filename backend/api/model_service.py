"""
model_service.py — Model Service for Flask API
Loads the trained model and provides prediction functions.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import sys
import shap

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import (
    ALL_INGREDIENTS, TOP_NOTES, MIDDLE_NOTES, BASE_NOTES,
    INGREDIENT_NOTE_MAP, FRAGRANCE_FAMILIES, APPLICATION_TYPES,
    LONGEVITY_RANGES
)
from feature_engineering import (
    create_derived_features, create_ingredient_counts,
    create_multihot_ingredients, encode_categoricals, get_feature_columns
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


class ModelService:
    def __init__(self):
        self.model_artifact = None
        self.model = None
        self.feature_cols = None
        self.transform = None
        self.metrics = None
        self.all_metrics = None
        self.optimal_ingredients = None
        self.ingredient_contributions = None
        self._load_model()
        self._load_results()

    def _load_model(self):
        """Load the best trained model from disk."""
        model_path = os.path.join(RESULTS_DIR, 'best_model.joblib')
        if not os.path.exists(model_path):
            print(f"[ModelService] WARNING: Model not found at {model_path}")
            return

        self.model_artifact = joblib.load(model_path)
        self.model = self.model_artifact['model']
        self.feature_cols = self.model_artifact['feature_columns']
        self.transform = self.model_artifact.get('transform', 'log')
        self.metrics = self.model_artifact.get('metrics', {})
        print(f"[ModelService] SUCCESS: Loaded model: {self.model_artifact.get('model_name', 'Unknown')}")

    def _load_results(self):
        """Load comparison results and optimization data."""
        # Model comparison
        comp_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        if os.path.exists(comp_path):
            df = pd.read_csv(comp_path)
            self.all_metrics = df.to_dict('records')

        # Optimal ingredients
        opt_path = os.path.join(RESULTS_DIR, 'optimal_ingredients.csv')
        if os.path.exists(opt_path):
            df = pd.read_csv(opt_path)
            self.optimal_ingredients = df.to_dict('records')

        # Ingredient contributions
        contrib_path = os.path.join(RESULTS_DIR, 'ingredient_contributions.json')
        if os.path.exists(contrib_path):
            with open(contrib_path, 'r') as f:
                self.ingredient_contributions = json.load(f)

    def is_ready(self):
        return self.model is not None

    def predict(self, input_data):
        """
        Predict scent duration from input parameters.

        input_data should be a dict with keys:
        - FragranceFamily, ApplicationType, Concentration_Pct
        - TopNotes_Pct, MiddleNotes_Pct, BaseNotes_Pct
        - Ingredients (comma-separated string or list)
        - AvgMolecularWeight, VolatilityScore, LogP, VaporPressure_Pa, BoilingPoint_C
        """
        if not self.is_ready():
            return {'error': 'Model not loaded'}

        # Build a single-row DataFrame
        row = {
            'FragranceID': 'INPUT',
            'FragranceFamily': input_data.get('FragranceFamily', 'Floral'),
            'ApplicationType': input_data.get('ApplicationType', 'Eau de Parfum'),
            'Concentration_Pct': float(input_data.get('Concentration_Pct', 10)),
            'TopNotes_Pct': float(input_data.get('TopNotes_Pct', 30)),
            'MiddleNotes_Pct': float(input_data.get('MiddleNotes_Pct', 30)),
            'BaseNotes_Pct': float(input_data.get('BaseNotes_Pct', 40)),
            'AvgMolecularWeight': int(input_data.get('AvgMolecularWeight', 189)),
            'VolatilityScore': float(input_data.get('VolatilityScore', 0.45)),
            'LogP': float(input_data.get('LogP', 3.67)),
            'VaporPressure_Pa': int(input_data.get('VaporPressure_Pa', 23)),
            'BoilingPoint_C': int(input_data.get('BoilingPoint_C', 266)),
            'ScentDuration_Hours': 0,
            'LongevityCategory': 'Short',
        }

        # Handle ingredients
        ingredients = input_data.get('Ingredients', '')
        if isinstance(ingredients, list):
            ingredients = ', '.join(ingredients)
        row['Ingredients'] = ingredients
        row['NumIngredients'] = len([i.strip() for i in ingredients.split(',') if i.strip()])

        df = pd.DataFrame([row])

        # Apply feature engineering
        df = create_derived_features(df)
        df = create_ingredient_counts(df)
        df = create_multihot_ingredients(df)
        df = encode_categoricals(df, use_onehot=True)

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_cols]

        # Predict
        y_pred_raw = self.model.predict(X)
        if self.transform == 'log':
            y_pred = float(np.expm1(y_pred_raw[0]))
        else:
            y_pred = float(y_pred_raw[0])

        # SHAP Local Explainability
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(X)[0]
        
        contributions = []
        for i, col in enumerate(self.feature_cols):
            val = float(sv[i])
            if abs(val) > 0.005:  # small threshold
                # Clean up feature name for UI
                name = col.replace('Ing_', '').replace('Family_', '').replace('App_', '').replace('_Pct', ' %')
                contributions.append({'feature': name, 'value': val})
                
        contributions.sort(key=lambda x: x['value'], reverse=True)
        positive = [c for c in contributions if c['value'] > 0][:3]
        
        contributions.sort(key=lambda x: x['value']) # most negative first
        negative = [c for c in contributions if c['value'] < 0][:3]

        y_pred = max(0.1, min(48, y_pred))

        # Assign category
        if y_pred < 3:
            category = 'Short'
        elif y_pred < 6:
            category = 'Medium'
        elif y_pred < 10:
            category = 'Long'
        else:
            category = 'Very Long'

        return {
            'predicted_duration_hours': round(y_pred, 2),
            'longevity_category': category,
            'model_used': self.model_artifact.get('model_name', 'Unknown'),
            'input_summary': {
                'family': row['FragranceFamily'],
                'application': row['ApplicationType'],
                'concentration': row['Concentration_Pct'],
                'num_ingredients': row['NumIngredients'],
            },
            'explainability': {
                'positive': positive,
                'negative': negative
            }
        }

    def get_model_info(self):
        """Return model performance metrics."""
        return {
            'best_model': self.model_artifact.get('model_name', 'Unknown') if self.model_artifact else None,
            'best_metrics': self.metrics,
            'all_metrics': self.all_metrics,
            'is_ready': self.is_ready(),
        }

    def get_ingredients(self):
        """Return ingredients grouped by note type."""
        return {
            'top_notes': TOP_NOTES,
            'middle_notes': MIDDLE_NOTES,
            'base_notes': BASE_NOTES,
            'all_ingredients': ALL_INGREDIENTS,
        }

    def get_options(self):
        """Return available fragrance families and application types."""
        return {
            'fragrance_families': FRAGRANCE_FAMILIES,
            'application_types': APPLICATION_TYPES,
        }

    def get_optimal_combinations(self, limit=15):
        """Return top optimal ingredient combinations."""
        if self.optimal_ingredients:
            return self.optimal_ingredients[:limit]
        return []

    def get_ingredient_contributions(self):
        """Return per-ingredient SHAP contributions."""
        return self.ingredient_contributions or {}
