"""
data_cleaning.py — Data Cleaning & Preprocessing Module
Handles duplicates, outliers, type validation, string normalization, and data quality reporting.
"""

import pandas as pd
import numpy as np
from data_loader import (
    parse_ingredients, ALL_INGREDIENTS, INGREDIENT_NOTE_MAP,
    LONGEVITY_RANGES, FRAGRANCE_FAMILIES, APPLICATION_TYPES
)


def detect_duplicates(df):
    """Detect and report duplicate rows (by FragranceID and by full composition)."""
    report = {}

    # Duplicate IDs
    dup_ids = df[df['FragranceID'].duplicated(keep=False)]
    report['duplicate_ids'] = len(dup_ids)
    if len(dup_ids) > 0:
        print(f"[Cleaning] ⚠ Found {len(dup_ids)} duplicate FragranceID rows")

    # Duplicate compositions (same features, different IDs)
    feature_cols = ['FragranceFamily', 'ApplicationType', 'Concentration_Pct',
                    'TopNotes_Pct', 'MiddleNotes_Pct', 'BaseNotes_Pct',
                    'Ingredients', 'AvgMolecularWeight', 'VolatilityScore',
                    'LogP', 'VaporPressure_Pa', 'BoilingPoint_C']
    dup_compositions = df[df.duplicated(subset=feature_cols, keep=False)]
    report['duplicate_compositions'] = len(dup_compositions)
    if len(dup_compositions) > 0:
        print(f"[Cleaning] ⚠ Found {len(dup_compositions)} rows with duplicate compositions")
    else:
        print("[Cleaning] ✓ No duplicate compositions found")

    return report


def validate_data_types(df):
    """Ensure correct data types for each column."""
    expected_types = {
        'FragranceID': 'object',
        'FragranceFamily': 'object',
        'ApplicationType': 'object',
        'Concentration_Pct': 'float64',
        'TopNotes_Pct': 'float64',
        'MiddleNotes_Pct': 'float64',
        'BaseNotes_Pct': 'float64',
        'Ingredients': 'object',
        'NumIngredients': 'int64',
        'AvgMolecularWeight': 'int64',
        'VolatilityScore': 'float64',
        'LogP': 'float64',
        'VaporPressure_Pa': 'int64',
        'BoilingPoint_C': 'int64',
        'ScentDuration_Hours': 'float64',
        'LongevityCategory': 'object'
    }

    type_issues = []
    for col, expected in expected_types.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if actual != expected:
                # Try to coerce numeric types
                if expected in ('float64', 'int64'):
                    try:
                        if expected == 'float64':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        else:
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                        type_issues.append(f"  {col}: coerced {actual} → {expected}")
                    except Exception as e:
                        type_issues.append(f"  {col}: FAILED to coerce {actual} → {expected}: {e}")
                else:
                    df[col] = df[col].astype(str)

    if type_issues:
        print("[Cleaning] Type coercions applied:")
        for issue in type_issues:
            print(issue)
    else:
        print("[Cleaning] ✓ All data types are correct")

    return df


def detect_outliers_iqr(df, columns=None, factor=1.5):
    """Detect outliers using IQR method. Returns a report dict."""
    if columns is None:
        columns = ['Concentration_Pct', 'AvgMolecularWeight', 'VolatilityScore',
                    'LogP', 'VaporPressure_Pa', 'BoilingPoint_C', 'ScentDuration_Hours']

    outlier_report = {}
    total_outliers = 0

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = outlier_mask.sum()
        outlier_report[col] = {
            'count': int(n_outliers),
            'lower_bound': round(lower, 2),
            'upper_bound': round(upper, 2),
            'min': round(df[col].min(), 2),
            'max': round(df[col].max(), 2)
        }
        total_outliers += n_outliers

    print(f"[Cleaning] Outlier detection (IQR × {factor}):")
    for col, info in outlier_report.items():
        status = f"  {col}: {info['count']} outliers"
        if info['count'] > 0:
            status += f" (range [{info['lower_bound']}, {info['upper_bound']}], data [{info['min']}, {info['max']}])"
        print(status)

    print(f"[Cleaning] Total outlier flags: {total_outliers} (NOT removed — tree models handle them)")

    return outlier_report


def normalize_ingredients(df):
    """Normalize ingredient strings: strip whitespace, validate against vocabulary."""
    cleaned_count = 0

    def clean_ingredient_str(ing_str):
        nonlocal cleaned_count
        if pd.isna(ing_str):
            return ing_str
        ingredients = parse_ingredients(ing_str)
        cleaned = []
        for ing in ingredients:
            ing = ing.strip()
            # Try case-insensitive match
            matched = None
            for known in ALL_INGREDIENTS:
                if ing.lower() == known.lower():
                    matched = known
                    break
            if matched:
                cleaned.append(matched)
            else:
                cleaned.append(ing)  # Keep as-is if unknown
        result = ', '.join(cleaned)
        if result != ing_str:
            cleaned_count += 1
        return result

    df['Ingredients'] = df['Ingredients'].apply(clean_ingredient_str)
    print(f"[Cleaning] ✓ Ingredient strings normalized ({cleaned_count} modified)")
    return df


def validate_longevity_consistency(df):
    """Check that LongevityCategory matches the ScentDuration_Hours ranges."""
    mismatches = 0
    for _, row in df.iterrows():
        hours = row['ScentDuration_Hours']
        cat = row['LongevityCategory']
        if cat in LONGEVITY_RANGES:
            lo, hi = LONGEVITY_RANGES[cat]
            if not (lo <= hours <= hi or (cat == 'Short' and hours < hi)):
                # Allow boundary edge cases
                pass

    # Simplified check based on data dictionary ranges
    conditions = [
        (df['LongevityCategory'] == 'Short') & (df['ScentDuration_Hours'] >= 3),
        (df['LongevityCategory'] == 'Medium') & ((df['ScentDuration_Hours'] < 3) | (df['ScentDuration_Hours'] > 6)),
        (df['LongevityCategory'] == 'Long') & ((df['ScentDuration_Hours'] < 6) | (df['ScentDuration_Hours'] > 10)),
        (df['LongevityCategory'] == 'Very Long') & (df['ScentDuration_Hours'] <= 10),
    ]
    for cond in conditions:
        mismatches += cond.sum()

    if mismatches > 0:
        print(f"[Cleaning] ⚠ {mismatches} rows have LongevityCategory mismatching ScentDuration_Hours ranges")
    else:
        print("[Cleaning] ✓ LongevityCategory is consistent with ScentDuration_Hours")

    return mismatches


def run_full_cleaning(df):
    """Run the complete cleaning pipeline. Returns cleaned DataFrame and quality report."""
    print("=" * 60)
    print("DATA CLEANING & PREPROCESSING REPORT")
    print("=" * 60)

    report = {}

    # 1. Duplicate detection
    report['duplicates'] = detect_duplicates(df)

    # 2. Type validation
    df = validate_data_types(df)

    # 3. Ingredient normalization
    df = normalize_ingredients(df)

    # 4. Outlier detection (flag only, don't remove for tree models)
    report['outliers'] = detect_outliers_iqr(df)

    # 5. Range validation
    note_sums = df['TopNotes_Pct'] + df['MiddleNotes_Pct'] + df['BaseNotes_Pct']
    bad_note_sums = ((note_sums < 99.5) | (note_sums > 100.5)).sum()
    print(f"[Cleaning] Note sum validation: {bad_note_sums} rows outside 99.5-100.5% range")
    report['bad_note_sums'] = int(bad_note_sums)

    # 6. Longevity consistency
    report['longevity_mismatches'] = int(validate_longevity_consistency(df))

    # 7. Final null check after cleaning
    nulls_after = df.isnull().sum().sum()
    print(f"[Cleaning] Final null count: {nulls_after}")
    report['nulls_after_cleaning'] = int(nulls_after)

    print("=" * 60)
    print("[Cleaning] ✅ Cleaning complete. DataFrame ready for feature engineering.")
    print("=" * 60)

    return df, report


if __name__ == '__main__':
    from data_loader import load_dataset
    df = load_dataset()
    df_clean, report = run_full_cleaning(df)
    print(f"\nCleaned shape: {df_clean.shape}")
