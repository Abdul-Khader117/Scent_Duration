"""
main.py — Main Pipeline Orchestrator
Runs the complete scent duration prediction pipeline end-to-end.
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress optuna logging
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    start_time = time.time()

    print("+" + "=" * 58 + "+")
    print("|   IFF Scent Duration Prediction -- Full Pipeline         |")
    print("+" + "=" * 58 + "+")
    print()

    # ── Phase 1: Data Loading ──
    from data_loader import load_dataset, validate_data
    df = load_dataset()
    validate_data(df)

    # ── Phase 2: Data Cleaning ──
    from data_cleaning import run_full_cleaning
    df_clean, clean_report = run_full_cleaning(df)

    # ── Phase 3: EDA ──
    from eda import run_full_eda
    run_full_eda(df_clean)

    # ── Phase 4: Feature Engineering ──
    from feature_engineering import build_feature_matrix
    df_engineered, feature_cols, X, y_reg, y_cls = build_feature_matrix(df_clean)

    # ── Phase 5: Model Training ──
    from model_training import run_full_training
    training_results = run_full_training(X, y_reg, y_cls, feature_cols, n_trials=80)

    # ── Phase 6: Explainability ──
    from explainability import run_full_explainability
    df_importance = run_full_explainability(training_results)

    # ── Phase 7: Ingredient Optimization ──
    from ingredient_optimizer import run_full_optimization
    optimization_results = run_full_optimization(df_clean, df_engineered, feature_cols, df_importance)

    # -- Final Summary --
    elapsed = time.time() - start_time
    print()
    print("+" + "=" * 58 + "+")
    print("|                    PIPELINE COMPLETE                     |")
    print("+" + "=" * 58 + "+")

    best_name = training_results['best_model_name']
    best_metrics = training_results['all_metrics']

    print("|  MODEL COMPARISON (Regression):                          |")
    print("+" + "=" * 58 + "+")
    print("|  {:<14s} {:>8s} {:>8s} {:>8s} {:>8s}  |".format(
        'Model', 'MAE(h)', 'RMSE(h)', 'R2', 'MAPE%'))
    print("|  " + "-" * 54 + "  |")
    for m in best_metrics:
        marker = " <<BEST>>" if m['Model'] == best_name else "         "
        print("|  {:<14s} {:>8.3f} {:>8.3f} {:>8.4f} {:>7.2f}{}|".format(
            m['Model'], m['MAE (hours)'], m['RMSE (hours)'], m['R²'], m['MAPE (%)'], marker))

    print("+" + "=" * 58 + "+")
    print(f"|  BEST Model: {best_name:<43s} |")
    print(f"|  Total Time: {elapsed:.1f}s{' ' * (43 - len(f'{elapsed:.1f}s'))}|")
    print("+" + "=" * 58 + "+")
    print()
    print("Output files:")
    print("  plots/                  -- EDA & SHAP visualizations")
    print("  results/model_comparison.csv")
    print("  results/optimal_ingredients.csv")
    print("  results/feature_importance.csv")
    print("  results/best_model.joblib")
    print("  results/classification_results.json")
    print()
    print("Next: Run the Flask API (python api/app.py) and React frontend (cd frontend && npm run dev)")


if __name__ == '__main__':
    main()
