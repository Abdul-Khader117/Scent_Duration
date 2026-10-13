"""
validate_predictions.py — Validate Model Predictions Against Ground Truth
Picks real samples from the training dataset, sends them through the prediction API,
and compares predicted vs actual scent duration.
"""

import requests
import pandas as pd
import numpy as np
import json
import sys
import os

API_BASE = "http://localhost:5000/api"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import parse_ingredients, TOP_NOTES, MIDDLE_NOTES, BASE_NOTES

def load_dataset():
    """Load the original dataset."""
    path = os.path.join(os.path.dirname(__file__), "ProjectB_Scent_Duration_Dataset.xlsx")
    df = pd.read_excel(path)
    return df

def predict_via_api(row):
    """Send a single sample to the Flask API and get the prediction."""
    ingredients = parse_ingredients(row['Ingredients'])
    
    payload = {
        "FragranceFamily": row["FragranceFamily"],
        "ApplicationType": row["ApplicationType"],
        "Concentration_Pct": float(row["Concentration_Pct"]),
        "TopNotes_Pct": float(row["TopNotes_Pct"]),
        "MiddleNotes_Pct": float(row["MiddleNotes_Pct"]),
        "BaseNotes_Pct": float(row["BaseNotes_Pct"]),
        "AvgMolecularWeight": int(row["AvgMolecularWeight"]),
        "VolatilityScore": float(row["VolatilityScore"]),
        "LogP": float(row["LogP"]),
        "VaporPressure_Pa": int(row["VaporPressure_Pa"]),
        "BoilingPoint_C": int(row["BoilingPoint_C"]),
        "Ingredients": ingredients,
    }
    
    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)
    if resp.status_code != 200:
        return None
    return resp.json()

def main():
    print("=" * 70)
    print("   ScentAI — PREDICTION VALIDATION AGAINST GROUND TRUTH")
    print("=" * 70)
    
    # 1. Load full dataset
    df = load_dataset()
    print(f"\n📦 Loaded dataset: {len(df)} samples")
    print(f"   Duration range: {df['ScentDuration_Hours'].min():.2f}h — {df['ScentDuration_Hours'].max():.2f}h")
    
    # 2. Check API is running
    try:
        resp = requests.get(f"{API_BASE}/model-info", timeout=5)
        model_info = resp.json()
        print(f"   Model loaded:  {model_info.get('best_model', 'Unknown')}")
    except Exception as e:
        print(f"\n❌ ERROR: Flask API is not reachable at {API_BASE}.")
        print(f"   Start it with: python api/app.py")
        return
    
    # 3. Pick diverse test samples
    # Stratified: pick samples from each longevity category + some edge cases
    np.random.seed(42)
    
    test_indices = []
    
    # Pick 3 from each category
    for cat in ['Short', 'Medium', 'Long', 'Very Long']:
        cat_rows = df[df['LongevityCategory'] == cat]
        if len(cat_rows) >= 3:
            test_indices.extend(cat_rows.sample(3, random_state=42).index.tolist())
        else:
            test_indices.extend(cat_rows.index.tolist())
    
    # Add the min duration sample, max duration sample, and a median sample
    test_indices.append(df['ScentDuration_Hours'].idxmin())
    test_indices.append(df['ScentDuration_Hours'].idxmax())
    median_val = df['ScentDuration_Hours'].median()
    test_indices.append((df['ScentDuration_Hours'] - median_val).abs().idxmin())
    
    # Remove duplicates
    test_indices = list(dict.fromkeys(test_indices))
    test_df = df.loc[test_indices].reset_index(drop=True)
    
    print(f"\n🧪 Testing {len(test_df)} diverse samples across all longevity categories...\n")
    
    # 4. Run predictions and compare
    results = []
    
    print(f"{'#':>3} | {'Actual':>8} | {'Predicted':>9} | {'Error':>7} | {'%Error':>7} | {'Cat(A)':>10} | {'Cat(P)':>10} | {'Match':>5}")
    print("-" * 85)
    
    for idx, row in test_df.iterrows():
        actual = row['ScentDuration_Hours']
        actual_cat = row['LongevityCategory']
        
        pred_result = predict_via_api(row)
        if pred_result is None:
            print(f"{idx+1:>3} | {'API ERR':>8} |")
            continue
        
        predicted = pred_result['predicted_duration_hours']
        pred_cat = pred_result['longevity_category']
        
        error = abs(predicted - actual)
        pct_error = (error / actual * 100) if actual > 0 else 0
        cat_match = "✅" if actual_cat == pred_cat else "❌"
        
        results.append({
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'pct_error': pct_error,
            'actual_cat': actual_cat,
            'pred_cat': pred_cat,
            'cat_match': actual_cat == pred_cat,
            'ingredients': row['Ingredients'][:50],
        })
        
        print(f"{len(results):>3} | {actual:>7.2f}h | {predicted:>8.2f}h | {error:>6.2f}h | {pct_error:>6.1f}% | {actual_cat:>10} | {pred_cat:>10} | {cat_match}")
    
    if not results:
        print("\n❌ No successful predictions. Check if the API is running.")
        return
    
    # 5. Compute aggregate metrics
    errors = [r['error'] for r in results]
    pct_errors = [r['pct_error'] for r in results]
    actuals = [r['actual'] for r in results]
    preds = [r['predicted'] for r in results]
    cat_matches = [r['cat_match'] for r in results]
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    mape = np.mean(pct_errors)
    
    # R² calculation
    ss_res = np.sum((np.array(actuals) - np.array(preds))**2)
    ss_tot = np.sum((np.array(actuals) - np.mean(actuals))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    cat_accuracy = sum(cat_matches) / len(cat_matches) * 100
    
    print("\n" + "=" * 70)
    print("   VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"   Samples tested:        {len(results)}")
    print(f"   MAE  (Avg Error):       {mae:.3f} hours ({mae*60:.1f} minutes)")
    print(f"   RMSE (Root Mean Sq):    {rmse:.3f} hours ({rmse*60:.1f} minutes)")
    print(f"   MAPE (Avg % Error):     {mape:.2f}%")
    print(f"   R² Score:               {r2:.4f}")
    print(f"   Category Accuracy:      {sum(cat_matches)}/{len(cat_matches)} ({cat_accuracy:.1f}%)")
    print()
    
    # Breakdown by category
    print("   📊 Error Breakdown by Category:")
    for cat in ['Short', 'Medium', 'Long', 'Very Long']:
        cat_results = [r for r in results if r['actual_cat'] == cat]
        if cat_results:
            cat_mae = np.mean([r['error'] for r in cat_results])
            cat_matches_ct = sum(1 for r in cat_results if r['cat_match'])
            print(f"      {cat:>10}: MAE = {cat_mae:.3f}h | Category Match: {cat_matches_ct}/{len(cat_results)}")
    
    print()
    
    # Find best and worst predictions
    best = min(results, key=lambda r: r['error'])
    worst = max(results, key=lambda r: r['error'])
    
    print(f"   🎯 Best prediction:  Actual={best['actual']:.2f}h, Predicted={best['predicted']:.2f}h, Error={best['error']:.3f}h")
    print(f"   ⚠️ Worst prediction: Actual={worst['actual']:.2f}h, Predicted={worst['predicted']:.2f}h, Error={worst['error']:.3f}h")
    
    print("\n" + "=" * 70)
    
    # Interpretation
    print("\n   📝 INTERPRETATION:")
    if mae < 1.0:
        print(f"   ✅ Excellent accuracy — predictions are off by only ~{mae*60:.0f} minutes on average")
    elif mae < 2.0:
        print(f"   ✅ Good accuracy — predictions are off by ~{mae*60:.0f} minutes on average")
    else:
        print(f"   ⚠️ Moderate accuracy — predictions are off by ~{mae*60:.0f} minutes on average")
    
    if r2 > 0.95:
        print(f"   ✅ R²={r2:.4f} — the model explains {r2*100:.1f}% of variance in scent duration")
    elif r2 > 0.85:
        print(f"   ✅ R²={r2:.4f} — strong predictive power ({r2*100:.1f}% variance explained)")
    
    if cat_accuracy >= 80:
        print(f"   ✅ Category accuracy {cat_accuracy:.0f}% — the model reliably classifies scent longevity")
    else:
        print(f"   ⚠️ Category accuracy {cat_accuracy:.0f}% — some borderline cases are misclassified")
    
    print()

if __name__ == "__main__":
    main()
