"""
explainability.py — SHAP Explainability Module
Generates SHAP-based feature importance analysis and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os
import warnings
warnings.filterwarnings('ignore')

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_shap_values(model, X, feature_cols, model_name='Model'):
    """Compute SHAP values for a trained model."""
    print(f"[SHAP] Computing SHAP values for {model_name}...")

    X_df = pd.DataFrame(X, columns=feature_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    print(f"[SHAP] ✓ Computed SHAP values: shape {shap_values.shape}")
    return explainer, shap_values, X_df


def plot_shap_summary(shap_values, X_df, model_name='Model'):
    """SHAP beeswarm summary plot."""
    ensure_dirs()
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_df, show=False, max_display=20)
    plt.title(f'SHAP Summary — {model_name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] ✓ Saved shap_summary.png")


def plot_shap_bar(shap_values, X_df, model_name='Model'):
    """SHAP bar plot for mean absolute feature importance."""
    ensure_dirs()
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_df, plot_type='bar', show=False, max_display=20)
    plt.title(f'Feature Importance (mean |SHAP|) — {model_name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] ✓ Saved shap_bar.png")


def plot_shap_dependence(shap_values, X_df, feature='Concentration_Pct', model_name='Model'):
    """SHAP dependence plot for a specific feature."""
    ensure_dirs()
    if feature not in X_df.columns:
        print(f"[SHAP] ⚠ Feature '{feature}' not found, skipping dependence plot")
        return
    plt.figure(figsize=(10, 7))
    shap.dependence_plot(feature, shap_values, X_df, show=False)
    plt.title(f'SHAP Dependence — {feature} ({model_name})', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'shap_dependence_{feature}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] ✓ Saved shap_dependence_{feature}.png")


def plot_shap_waterfall(explainer, shap_values, X_df, idx=0, model_name='Model'):
    """SHAP waterfall plot for a single prediction."""
    ensure_dirs()
    plt.figure(figsize=(12, 8))
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_df.iloc[idx],
        feature_names=X_df.columns.tolist()
    )
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(f'Prediction Breakdown — Sample #{idx} ({model_name})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'shap_waterfall_sample{idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] ✓ Saved shap_waterfall_sample{idx}.png")


def compare_feature_importance(all_models, X, feature_cols):
    """Compare feature importance rankings across all 3 models."""
    ensure_dirs()

    importance_data = {}
    for name, model_info in all_models.items():
        model = model_info['model']
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(pd.DataFrame(X, columns=feature_cols))
        mean_abs_shap = np.abs(sv).mean(axis=0)
        importance_data[name] = mean_abs_shap

    # Create comparison DataFrame
    df_importance = pd.DataFrame(importance_data, index=feature_cols)
    df_importance.columns = [n.capitalize() for n in df_importance.columns]

    # Normalize to relative importance
    for col in df_importance.columns:
        df_importance[col] = df_importance[col] / df_importance[col].sum() * 100

    # Sort by average importance
    df_importance['Average'] = df_importance.mean(axis=1)
    df_importance = df_importance.sort_values('Average', ascending=False)

    # Save
    df_importance.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'))
    print(f"[SHAP] ✓ Saved feature_importance.csv")

    # Plot top 20
    top_n = min(20, len(df_importance))
    top_features = df_importance.head(top_n).drop(columns='Average')

    fig, ax = plt.subplots(figsize=(12, 8))
    top_features.plot(kind='barh', ax=ax, width=0.75)
    ax.set_xlabel('Relative Importance (%)', fontsize=12)
    ax.set_title('Feature Importance Comparison Across Models', fontsize=15, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(title='Model', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'feature_importance_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] ✓ Saved feature_importance_comparison.png")

    return df_importance


def run_full_explainability(training_results):
    """Run the complete SHAP analysis pipeline."""
    print("=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    best_model = training_results['best_model']
    best_name = training_results['best_model_name']
    X_test = training_results['X_test']
    feature_cols = training_results['feature_cols']

    # Compute SHAP for best model
    explainer, shap_values, X_df = compute_shap_values(best_model, X_test, feature_cols, best_name)

    # Generate plots
    plot_shap_summary(shap_values, X_df, best_name)
    plot_shap_bar(shap_values, X_df, best_name)
    plot_shap_dependence(shap_values, X_df, 'Concentration_Pct', best_name)

    # Waterfall for first test sample
    plot_shap_waterfall(explainer, shap_values, X_df, idx=0, model_name=best_name)

    # Compare across all models
    all_models = {
        name.lower().replace(' ', '_'): info
        for name, info in training_results['all_models'].items()
    }
    # Restructure for the compare function
    all_models_info = {}
    for key, model_obj in all_models.items():
        all_models_info[key] = {'model': model_obj}

    X_all = np.vstack([training_results['X_train'], training_results['X_test']])
    df_importance = compare_feature_importance(all_models_info, X_all, feature_cols)

    print("=" * 60)
    print("[SHAP] ✅ Explainability analysis complete")
    print("=" * 60)

    return df_importance
