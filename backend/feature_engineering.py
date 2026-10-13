"""
feature_engineering.py — Feature Engineering Module
Creates derived features, encodes categoricals, builds multi-hot ingredient vectors,
and prepares the final feature matrix for modeling.
"""

import pandas as pd
import numpy as np
from data_loader import (
    parse_ingredients, ALL_INGREDIENTS, INGREDIENT_NOTE_MAP,
    TOP_NOTES, MIDDLE_NOTES, BASE_NOTES,
    FRAGRANCE_FAMILIES, APPLICATION_TYPES
)


def create_derived_features(df):
    """Create domain-informed derived features from raw columns."""
    df = df.copy()

    # Ratio features
    df['Base_to_Top_Ratio'] = df['BaseNotes_Pct'] / df['TopNotes_Pct'].clip(lower=0.1)
    df['Base_to_Total_Volatile'] = df['BaseNotes_Pct'] / (df['TopNotes_Pct'] + df['MiddleNotes_Pct']).clip(lower=0.1)

    # Interaction features
    df['Volatility_x_VaporPressure'] = df['VolatilityScore'] * df['VaporPressure_Pa']
    df['LogP_x_Concentration'] = df['LogP'] * df['Concentration_Pct']
    df['BoilingPoint_x_MolWeight'] = df['BoilingPoint_C'] * df['AvgMolecularWeight']
    df['Concentration_x_BaseNotes'] = df['Concentration_Pct'] * df['BaseNotes_Pct']

    # Normalized complexity
    df['Ingredient_Richness'] = df['NumIngredients'] / 9.0  # max is 9

    print(f"[FeatureEng] ✓ Created 7 derived features")
    return df


def create_ingredient_counts(df):
    """Count ingredients by note type from parsed ingredient lists."""
    df = df.copy()

    top_counts = []
    mid_counts = []
    base_counts = []

    for _, row in df.iterrows():
        ingredients = parse_ingredients(row['Ingredients'])
        top_c = sum(1 for ing in ingredients if ing in TOP_NOTES)
        mid_c = sum(1 for ing in ingredients if ing in MIDDLE_NOTES)
        base_c = sum(1 for ing in ingredients if ing in BASE_NOTES)
        top_counts.append(top_c)
        mid_counts.append(mid_c)
        base_counts.append(base_c)

    df['TopNote_Count'] = top_counts
    df['MiddleNote_Count'] = mid_counts
    df['BaseNote_Count'] = base_counts

    print(f"[FeatureEng] ✓ Created note count features (Top/Middle/Base)")
    return df


def create_multihot_ingredients(df):
    """Create 30 binary columns for each ingredient presence."""
    df = df.copy()

    for ing in ALL_INGREDIENTS:
        col_name = f'has_{ing.replace(" ", "_").replace("-", "_")}'
        df[col_name] = df['Ingredients'].apply(
            lambda x: 1 if ing in parse_ingredients(x) else 0
        )

    print(f"[FeatureEng] ✓ Created {len(ALL_INGREDIENTS)} multi-hot ingredient columns")
    return df


def encode_categoricals(df, use_onehot=True):
    """Encode FragranceFamily and ApplicationType."""
    df = df.copy()

    if use_onehot:
        # One-hot encode
        family_dummies = pd.get_dummies(df['FragranceFamily'], prefix='Family').astype(int)
        app_dummies = pd.get_dummies(df['ApplicationType'], prefix='App').astype(int)
        df = pd.concat([df, family_dummies, app_dummies], axis=1)
        print(f"[FeatureEng] ✓ One-hot encoded FragranceFamily ({len(family_dummies.columns)} cols) "
              f"and ApplicationType ({len(app_dummies.columns)} cols)")
    else:
        # Label encode (for LightGBM native categoricals)
        from sklearn.preprocessing import LabelEncoder
        le_family = LabelEncoder()
        le_app = LabelEncoder()
        df['Family_encoded'] = le_family.fit_transform(df['FragranceFamily'])
        df['App_encoded'] = le_app.fit_transform(df['ApplicationType'])
        print(f"[FeatureEng] ✓ Label encoded FragranceFamily and ApplicationType")

    return df


def get_feature_columns(df):
    """Return the list of feature column names for modeling (excludes ID, target, raw categoricals)."""
    exclude_cols = {
        'FragranceID', 'FragranceFamily', 'ApplicationType',
        'Ingredients', 'ScentDuration_Hours', 'LongevityCategory'
    }
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def build_feature_matrix(df):
    """Run the complete feature engineering pipeline.
    Returns (df_engineered, feature_columns, X, y_regression, y_classification).
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Step 1: Derived features
    df = create_derived_features(df)

    # Step 2: Note counts
    df = create_ingredient_counts(df)

    # Step 3: Multi-hot ingredients
    df = create_multihot_ingredients(df)

    # Step 4: Encode categoricals
    df = encode_categoricals(df, use_onehot=True)

    # Get feature columns
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y_reg = df['ScentDuration_Hours'].values
    y_cls = df['LongevityCategory'].values

    print(f"\n[FeatureEng] Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"[FeatureEng] Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:3d}. {col}")

    print("=" * 60)
    print("[FeatureEng] ✅ Feature engineering complete")
    print("=" * 60)

    return df, feature_cols, X, y_reg, y_cls


if __name__ == '__main__':
    from data_loader import load_dataset
    from data_cleaning import run_full_cleaning

    df = load_dataset()
    df, _ = run_full_cleaning(df)
    df_eng, feature_cols, X, y_reg, y_cls = build_feature_matrix(df)
    print(f"\nX shape: {X.shape}")
    print(f"y_reg range: [{y_reg.min():.1f}, {y_reg.max():.1f}]")
    print(f"y_cls classes: {np.unique(y_cls)}")
