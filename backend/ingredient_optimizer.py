"""
ingredient_optimizer.py — Ingredient Optimization Module
Finds optimal ingredient combinations for maximum scent longevity.
"""

import numpy as np
import pandas as pd
import os
import joblib
import json
from itertools import combinations
from data_loader import (
    ALL_INGREDIENTS, TOP_NOTES, MIDDLE_NOTES, BASE_NOTES,
    INGREDIENT_NOTE_MAP, FRAGRANCE_FAMILIES, APPLICATION_TYPES
)
from feature_engineering import (
    create_derived_features, create_ingredient_counts,
    create_multihot_ingredients, encode_categoricals, get_feature_columns
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_ingredient_shap_contributions(df_importance):
    """Extract per-ingredient SHAP contributions from feature importance data."""
    ingredient_contribs = {}
    for ing in ALL_INGREDIENTS:
        col_name = f'has_{ing.replace(" ", "_").replace("-", "_")}'
        if col_name in df_importance.index:
            avg_importance = df_importance.loc[col_name, 'Average']
            ingredient_contribs[ing] = round(avg_importance, 4)
    return dict(sorted(ingredient_contribs.items(), key=lambda x: x[1], reverse=True))


def analyze_top_combinations_from_data(df_original, top_n=20):
    """Analyze actual ingredient combinations in the dataset ranked by duration."""
    results = []
    for _, row in df_original.iterrows():
        ingredients = [ing.strip() for ing in str(row['Ingredients']).split(',')]
        ingredients_sorted = tuple(sorted(ingredients))
        results.append({
            'Ingredients': ', '.join(ingredients_sorted),
            'NumIngredients': len(ingredients),
            'Duration_Hours': row['ScentDuration_Hours'],
            'Concentration_Pct': row['Concentration_Pct'],
            'LongevityCategory': row['LongevityCategory'],
            'FragranceFamily': row['FragranceFamily'],
            'ApplicationType': row['ApplicationType'],
        })

    df_results = pd.DataFrame(results)
    # Group by ingredient combo and get mean duration
    combo_stats = df_results.groupby('Ingredients').agg({
        'Duration_Hours': ['mean', 'max', 'count'],
        'Concentration_Pct': 'mean',
    }).reset_index()
    combo_stats.columns = ['Ingredients', 'Avg_Duration', 'Max_Duration', 'Count', 'Avg_Concentration']
    combo_stats = combo_stats.sort_values('Avg_Duration', ascending=False)

    return combo_stats.head(top_n)


def predict_optimal_combinations(model_artifact, df_engineered, feature_cols,
                                  target_family=None, target_app=None,
                                  min_ingredients=4, max_ingredients=7,
                                  top_n=15):
    """Use the trained model to predict duration for synthetic ingredient combinations.
    Generates combinations and predicts which ones achieve maximum longevity.
    """
    model = model_artifact['model']
    transform = model_artifact.get('transform', 'log')

    # Get median values for numerical features as baseline
    baseline = {}
    for col in ['Concentration_Pct', 'TopNotes_Pct', 'MiddleNotes_Pct', 'BaseNotes_Pct',
                'NumIngredients', 'AvgMolecularWeight', 'VolatilityScore',
                'LogP', 'VaporPressure_Pa', 'BoilingPoint_C']:
        if col in df_engineered.columns:
            baseline[col] = df_engineered[col].median()

    # Use high concentration for longevity optimization
    baseline['Concentration_Pct'] = 15.0  # Eau de Parfum range

    # Generate ingredient combinations
    all_combos = []
    for n in range(min_ingredients, max_ingredients + 1):
        # Sample a subset of combinations to keep it tractable
        from itertools import combinations as comb_gen
        for combo in comb_gen(ALL_INGREDIENTS, n):
            all_combos.append(combo)

    print(f"[Optimizer] Generated {len(all_combos)} ingredient combinations")

    # Build synthetic samples
    results = []
    batch_size = 1000
    for batch_start in range(0, len(all_combos), batch_size):
        batch = all_combos[batch_start:batch_start + batch_size]
        rows = []
        for combo in batch:
            row = baseline.copy()
            row['Ingredients'] = ', '.join(combo)
            row['NumIngredients'] = len(combo)
            row['FragranceFamily'] = target_family or 'Floral'
            row['ApplicationType'] = target_app or 'Eau de Parfum'
            row['FragranceID'] = 'SYNTH'
            row['ScentDuration_Hours'] = 0
            row['LongevityCategory'] = 'Short'

            # Calculate note composition based on ingredient types
            top_count = sum(1 for ing in combo if ing in TOP_NOTES)
            mid_count = sum(1 for ing in combo if ing in MIDDLE_NOTES)
            base_count = sum(1 for ing in combo if ing in BASE_NOTES)
            total = max(top_count + mid_count + base_count, 1)

            row['TopNotes_Pct'] = round(25.0 * top_count / total + 20, 1)
            row['MiddleNotes_Pct'] = round(25.0 * mid_count / total + 20, 1)
            row['BaseNotes_Pct'] = round(100 - row['TopNotes_Pct'] - row['MiddleNotes_Pct'], 1)

            # Adjust chemistry based on composition
            row['VolatilityScore'] = max(0.28, 0.5 - 0.03 * base_count + 0.02 * top_count)
            row['BoilingPoint_C'] = int(250 + 5 * base_count - 3 * top_count)

            rows.append(row)

        df_synth = pd.DataFrame(rows)

        # Apply feature engineering
        df_synth = create_derived_features(df_synth)
        df_synth = create_ingredient_counts(df_synth)
        df_synth = create_multihot_ingredients(df_synth)
        df_synth = encode_categoricals(df_synth, use_onehot=True)

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df_synth.columns:
                df_synth[col] = 0

        X_synth = df_synth[feature_cols].values

        # Predict
        y_pred_raw = model.predict(X_synth)
        if transform == 'log':
            y_pred = np.expm1(y_pred_raw)
        else:
            y_pred = y_pred_raw
        y_pred = np.clip(y_pred, 0.1, 48)

        for combo, pred in zip(batch, y_pred):
            top_c = sum(1 for ing in combo if ing in TOP_NOTES)
            mid_c = sum(1 for ing in combo if ing in MIDDLE_NOTES)
            base_c = sum(1 for ing in combo if ing in BASE_NOTES)
            results.append({
                'Ingredients': ', '.join(combo),
                'NumIngredients': len(combo),
                'TopNote_Count': top_c,
                'MiddleNote_Count': mid_c,
                'BaseNote_Count': base_c,
                'Predicted_Duration_Hours': round(float(pred), 2),
            })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Predicted_Duration_Hours', ascending=False)

    # Assign longevity category
    def assign_category(hours):
        if hours < 3:
            return 'Short'
        elif hours < 6:
            return 'Medium'
        elif hours < 10:
            return 'Long'
        else:
            return 'Very Long'

    df_results['Predicted_Category'] = df_results['Predicted_Duration_Hours'].apply(assign_category)

    return df_results.head(top_n)


def run_full_optimization(df_original, df_engineered, feature_cols, df_importance=None):
    """Run the complete ingredient optimization pipeline."""
    ensure_dirs()
    print("=" * 60)
    print("INGREDIENT OPTIMIZATION")
    print("=" * 60)

    # 1. Analyze best combos from actual data
    print("\n[Optimizer] Analyzing top ingredient combinations from dataset...")
    top_data_combos = analyze_top_combinations_from_data(df_original, top_n=20)
    top_data_combos.to_csv(os.path.join(RESULTS_DIR, 'top_data_combinations.csv'), index=False)
    print(f"[Optimizer] ✓ Saved top_data_combinations.csv")

    print("\n  Top 5 Data Combinations:")
    for _, row in top_data_combos.head(5).iterrows():
        print(f"  {row['Avg_Duration']:.1f}h avg | {row['Ingredients']}")

    # 2. Ingredient SHAP contributions
    if df_importance is not None:
        print("\n[Optimizer] Extracting ingredient SHAP contributions...")
        ing_contribs = get_ingredient_shap_contributions(df_importance)
        with open(os.path.join(RESULTS_DIR, 'ingredient_contributions.json'), 'w') as f:
            json.dump(ing_contribs, f, indent=2)
        print(f"[Optimizer] ✓ Saved ingredient_contributions.json")
        print("\n  Top 10 Ingredients by SHAP contribution:")
        for i, (ing, score) in enumerate(list(ing_contribs.items())[:10]):
            note_type = INGREDIENT_NOTE_MAP.get(ing, '?')
            print(f"  {i+1:2d}. {ing:15s} ({note_type:6s}) — importance: {score:.4f}")

    # 3. Model-based optimization
    model_path = os.path.join(RESULTS_DIR, 'best_model.joblib')
    if os.path.exists(model_path):
        print("\n[Optimizer] Running model-based optimization...")
        model_artifact = joblib.load(model_path)

        optimal = predict_optimal_combinations(
            model_artifact, df_engineered, feature_cols,
            min_ingredients=4, max_ingredients=6,
            top_n=25
        )
        optimal.to_csv(os.path.join(RESULTS_DIR, 'optimal_ingredients.csv'), index=False)
        print(f"[Optimizer] ✓ Saved optimal_ingredients.csv")

        print("\n  Top 10 Predicted Optimal Combinations:")
        for _, row in optimal.head(10).iterrows():
            print(f"  {row['Predicted_Duration_Hours']:5.1f}h ({row['Predicted_Category']:9s}) | "
                  f"T:{row['TopNote_Count']} M:{row['MiddleNote_Count']} B:{row['BaseNote_Count']} | "
                  f"{row['Ingredients']}")
    else:
        print("[Optimizer] ⚠ No trained model found, skipping model-based optimization")

    print("=" * 60)
    print("[Optimizer] ✅ Ingredient optimization complete")
    print("=" * 60)

    return {
        'top_data_combos': top_data_combos,
    }
