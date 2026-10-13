"""
model_training.py — Model Training, Tuning & Comparison Module
Trains XGBoost, LightGBM, and Random Forest for both regression and classification.
Uses Optuna for hyperparameter tuning and generates comparison reports.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def split_data(X, y_reg, y_cls, test_size=0.2, random_state=42):
    """Split data into train/test with stratification by longevity category."""
    le = LabelEncoder()
    y_cls_encoded = le.fit_transform(y_cls)

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls_encoded,
        test_size=test_size, random_state=random_state, stratify=y_cls_encoded
    )

    print(f"[Training] Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, le


# ──────────────────────────────────────────────────────────────────────────────
# REGRESSION MODELS
# ──────────────────────────────────────────────────────────────────────────────

def tune_xgboost_regression(X_train, y_train, n_trials=80):
    """Tune XGBoost regressor with Optuna."""
    # Log-transform target
    y_log = np.log1p(y_train)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'verbosity': 0,
        }
        model = xgb.XGBRegressor(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Stratify by binning the log target
        bins = pd.qcut(y_log, q=5, labels=False, duplicates='drop')
        scores = cross_val_score(model, X_train, y_log, cv=cv.split(X_train, bins),
                                 scoring='neg_mean_absolute_error')
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['verbosity'] = 0
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_log)

    print(f"[Training] XGBoost best MAE (CV, log-space): {study.best_value:.4f}")
    return model, best_params, 'log'


def tune_lightgbm_regression(X_train, y_train, n_trials=80):
    """Tune LightGBM regressor with Optuna."""
    y_log = np.log1p(y_train)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbosity': -1,
        }
        model = lgb.LGBMRegressor(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        bins = pd.qcut(y_log, q=5, labels=False, duplicates='drop')
        scores = cross_val_score(model, X_train, y_log, cv=cv.split(X_train, bins),
                                 scoring='neg_mean_absolute_error')
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['verbosity'] = -1
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_log)

    print(f"[Training] LightGBM best MAE (CV, log-space): {study.best_value:.4f}")
    return model, best_params, 'log'


def tune_random_forest_regression(X_train, y_train, n_trials=60):
    """Tune Random Forest regressor with Optuna."""
    y_log = np.log1p(y_train)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
        }
        model = RandomForestRegressor(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        bins = pd.qcut(y_log, q=5, labels=False, duplicates='drop')
        scores = cross_val_score(model, X_train, y_log, cv=cv.split(X_train, bins),
                                 scoring='neg_mean_absolute_error')
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params['random_state'] = 42
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_log)

    print(f"[Training] RandomForest best MAE (CV, log-space): {study.best_value:.4f}")
    return model, best_params, 'log'


def evaluate_regression(model, X_test, y_test, model_name, transform='log'):
    """Evaluate regression model on test set. Returns metrics dict."""
    y_pred_raw = model.predict(X_test)

    if transform == 'log':
        y_pred = np.expm1(y_pred_raw)
    else:
        y_pred = y_pred_raw

    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 0.1, 48)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE (avoid division by zero)
    nonzero = y_test > 0.1
    mape = np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100

    metrics = {
        'Model': model_name,
        'MAE (hours)': round(mae, 3),
        'RMSE (hours)': round(rmse, 3),
        'R²': round(r2, 4),
        'MAPE (%)': round(mape, 2)
    }

    print(f"\n[Eval] {model_name} Regression Results:")
    print(f"  MAE  = {mae:.3f} hours")
    print(f"  RMSE = {rmse:.3f} hours")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAPE = {mape:.2f}%")

    return metrics, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION MODELS
# ──────────────────────────────────────────────────────────────────────────────

def train_classification_models(X_train, y_cls_train, X_test, y_cls_test, le, best_reg_params):
    """Train classification variants of all 3 models using best regression params as hints."""
    cls_results = {}

    # XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=best_reg_params.get('xgboost', {}).get('n_estimators', 300),
        max_depth=best_reg_params.get('xgboost', {}).get('max_depth', 6),
        learning_rate=best_reg_params.get('xgboost', {}).get('learning_rate', 0.1),
        random_state=42, verbosity=0,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train, y_cls_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    cls_results['XGBoost'] = {
        'accuracy': round(accuracy_score(y_cls_test, y_pred_xgb), 4),
        'f1_weighted': round(f1_score(y_cls_test, y_pred_xgb, average='weighted'), 4),
        'report': classification_report(y_cls_test, y_pred_xgb,
                                        target_names=le.classes_, output_dict=True),
        'confusion_matrix': confusion_matrix(y_cls_test, y_pred_xgb).tolist()
    }

    # LightGBM Classifier
    if HAS_LIGHTGBM and 'lightgbm' in best_reg_params:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=best_reg_params.get('lightgbm', {}).get('n_estimators', 300),
            num_leaves=best_reg_params.get('lightgbm', {}).get('num_leaves', 31),
            learning_rate=best_reg_params.get('lightgbm', {}).get('learning_rate', 0.1),
            random_state=42, verbosity=-1
        )
        lgb_clf.fit(X_train, y_cls_train)
        y_pred_lgb = lgb_clf.predict(X_test)
        cls_results['LightGBM'] = {
            'accuracy': round(accuracy_score(y_cls_test, y_pred_lgb), 4),
            'f1_weighted': round(f1_score(y_cls_test, y_pred_lgb, average='weighted'), 4),
            'report': classification_report(y_cls_test, y_pred_lgb,
                                            target_names=le.classes_, output_dict=True),
            'confusion_matrix': confusion_matrix(y_cls_test, y_pred_lgb).tolist()
        }

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=best_reg_params.get('random_forest', {}).get('n_estimators', 300),
        max_depth=best_reg_params.get('random_forest', {}).get('max_depth', 15),
        random_state=42
    )
    rf_clf.fit(X_train, y_cls_train)
    y_pred_rf = rf_clf.predict(X_test)
    cls_results['RandomForest'] = {
        'accuracy': round(accuracy_score(y_cls_test, y_pred_rf), 4),
        'f1_weighted': round(f1_score(y_cls_test, y_pred_rf, average='weighted'), 4),
        'report': classification_report(y_cls_test, y_pred_rf,
                                        target_names=le.classes_, output_dict=True),
        'confusion_matrix': confusion_matrix(y_cls_test, y_pred_rf).tolist()
    }

    print("\n[Classification] Results:")
    for name, res in cls_results.items():
        print(f"  {name}: Accuracy={res['accuracy']}, F1(weighted)={res['f1_weighted']}")

    return cls_results


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON & VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(all_metrics):
    """Create comparison bar chart for all 3 models."""
    ensure_dirs()
    df_metrics = pd.DataFrame(all_metrics)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_to_plot = ['MAE (hours)', 'RMSE (hours)', 'R²', 'MAPE (%)']
    colors = ['#e74c3c', '#2196F3', '#4CAF50'][:len(df_metrics)]

    for i, metric in enumerate(metrics_to_plot):
        bars = axes[i].bar(df_metrics['Model'], df_metrics[metric], color=colors,
                           edgecolor='white', linewidth=1.5)
        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric)
        for bar, val in zip(bars, df_metrics[metric]):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val}', ha='center', fontsize=10, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=20)

    plt.suptitle('Model Comparison — Regression Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Training] ✓ Saved model_comparison.png")


def plot_residuals(y_test, predictions_dict):
    """Plot residual distributions for all models."""
    ensure_dirs()
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * max(n_models, 1), 5))
    axes = np.atleast_1d(axes)
    colors = ['#e74c3c', '#2196F3', '#4CAF50']

    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        residuals = y_test - y_pred
        axes[i].scatter(y_pred, residuals, alpha=0.5, color=colors[i % len(colors)], s=30, edgecolors='white')
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[i].set_xlabel('Predicted Duration (h)', fontsize=11)
        axes[i].set_ylabel('Residual (h)', fontsize=11)
        axes[i].set_title(f'{name} Residuals', fontsize=13, fontweight='bold')

    plt.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'residuals.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Training] ✓ Saved residuals.png")


def plot_actual_vs_predicted(y_test, predictions_dict):
    """Actual vs Predicted scatter for all models."""
    ensure_dirs()
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * max(n_models, 1), 5))
    axes = np.atleast_1d(axes)
    colors = ['#e74c3c', '#2196F3', '#4CAF50']

    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        axes[i].scatter(y_test, y_pred, alpha=0.5, color=colors[i % len(colors)], s=30, edgecolors='white')
        max_val = max(y_test.max(), y_pred.max()) + 1
        axes[i].plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect fit')
        axes[i].set_xlabel('Actual Duration (h)', fontsize=11)
        axes[i].set_ylabel('Predicted Duration (h)', fontsize=11)
        axes[i].set_title(f'{name}', fontsize=13, fontweight='bold')
        axes[i].legend()

    plt.suptitle('Actual vs. Predicted Duration', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Training] ✓ Saved actual_vs_predicted.png")


def plot_classification_confusion(cls_results, le):
    """Plot confusion matrices for classification models."""
    ensure_dirs()
    n_models = len(cls_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * max(n_models, 1), 5))
    axes = np.atleast_1d(axes)

    for i, (name, res) in enumerate(cls_results.items()):
        cm = np.array(res['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=le.classes_, yticklabels=le.classes_)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        acc = res['accuracy']
        axes[i].set_title(f'{name} (Acc={acc:.2%})', fontsize=13, fontweight='bold')

    plt.suptitle('Classification Confusion Matrices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'classification_confusion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Training] ✓ Saved classification_confusion.png")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_full_training(X, y_reg, y_cls, feature_cols, n_trials=80):
    """Run the complete model training pipeline. Returns best model and comparison results."""
    ensure_dirs()
    print("=" * 60)
    print("MODEL TRAINING & COMPARISON")
    print("=" * 60)

    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, le = \
        split_data(X, y_reg, y_cls)

    # ── Train & Tune Regression Models ──
    print("\n" + "─" * 40)
    print("Training XGBoost...")
    print("─" * 40)
    xgb_model, xgb_params, xgb_transform = tune_xgboost_regression(X_train, y_reg_train, n_trials)

    all_metrics = []
    predictions_dict = {}
    models = {}
    best_reg_params = {}

    xgb_metrics, xgb_pred = evaluate_regression(xgb_model, X_test, y_reg_test, 'XGBoost', xgb_transform)
    all_metrics.append(xgb_metrics)
    predictions_dict['XGBoost'] = xgb_pred
    models['XGBoost'] = xgb_model
    best_reg_params['xgboost'] = xgb_params

    if HAS_LIGHTGBM:
        print("\n" + "─" * 40)
        print("Training LightGBM...")
        print("─" * 40)
        lgb_model, lgb_params, lgb_transform = tune_lightgbm_regression(X_train, y_reg_train, n_trials)
        lgb_metrics, lgb_pred = evaluate_regression(lgb_model, X_test, y_reg_test, 'LightGBM', lgb_transform)
        all_metrics.append(lgb_metrics)
        predictions_dict['LightGBM'] = lgb_pred
        models['LightGBM'] = lgb_model
        best_reg_params['lightgbm'] = lgb_params

    print("\n" + "─" * 40)
    print("Training Random Forest...")
    print("─" * 40)
    rf_model, rf_params, rf_transform = tune_random_forest_regression(X_train, y_reg_train, n_trials=min(n_trials, 60))
    rf_metrics, rf_pred = evaluate_regression(rf_model, X_test, y_reg_test, 'RandomForest', rf_transform)
    all_metrics.append(rf_metrics)
    predictions_dict['RandomForest'] = rf_pred
    models['RandomForest'] = rf_model
    best_reg_params['random_forest'] = rf_params

    # ── Train Classification Models ──
    print("\n" + "─" * 40)
    print("CLASSIFICATION TRAINING")
    print("─" * 40)
    cls_results = train_classification_models(X_train, y_cls_train, X_test, y_cls_test, le, best_reg_params)

    # ── Visualizations ──
    print("\n" + "─" * 40)
    print("GENERATING COMPARISON PLOTS")
    print("─" * 40)
    plot_model_comparison(all_metrics)
    plot_residuals(y_reg_test, predictions_dict)
    plot_actual_vs_predicted(y_reg_test, predictions_dict)
    plot_classification_confusion(cls_results, le)

    # ── Select Best Model ──
    best_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['MAE (hours)'])
    best_model_name = all_metrics[best_idx]['Model']
    best_model = models[best_model_name]

    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   MAE = {all_metrics[best_idx]['MAE (hours)']} hours")
    print(f"   R²  = {all_metrics[best_idx]['R²']}")
    print("=" * 60)

    # ── Save Results ──
    df_comparison = pd.DataFrame(all_metrics)
    df_comparison.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    print(f"[Training] ✓ Saved model_comparison.csv")

    # Save best model
    model_artifact = {
        'model': best_model,
        'model_name': best_model_name,
        'params': best_reg_params[best_model_name.lower().replace(' ', '_')
                                   if best_model_name != 'RandomForest' else 'random_forest'],
        'transform': 'log',
        'feature_columns': feature_cols,
        'label_encoder': le,
        'metrics': all_metrics[best_idx],
    }
    joblib.dump(model_artifact, os.path.join(RESULTS_DIR, 'best_model.joblib'))
    print(f"[Training] ✓ Saved best_model.joblib")

    # Save all models
    all_models_artifact = {
        'xgboost': {'model': xgb_model, 'params': xgb_params, 'transform': 'log'},
        'random_forest': {'model': rf_model, 'params': rf_params, 'transform': 'log'},
    }
    if HAS_LIGHTGBM:
        all_models_artifact['lightgbm'] = {'model': lgb_model, 'params': lgb_params, 'transform': 'log'}
    joblib.dump(all_models_artifact, os.path.join(RESULTS_DIR, 'all_models.joblib'))
    print(f"[Training] ✓ Saved all_models.joblib")

    # Save classification results
    with open(os.path.join(RESULTS_DIR, 'classification_results.json'), 'w') as f:
        json.dump(cls_results, f, indent=2)
    print(f"[Training] ✓ Saved classification_results.json")

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'all_metrics': all_metrics,
        'all_models': models,
        'cls_results': cls_results,
        'label_encoder': le,
        'X_train': X_train,
        'X_test': X_test,
        'y_reg_train': y_reg_train,
        'y_reg_test': y_reg_test,
        'y_cls_test': y_cls_test,
        'predictions': predictions_dict,
        'feature_cols': feature_cols,
    }
