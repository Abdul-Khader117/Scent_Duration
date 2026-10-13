"""
data_loader.py — Data Loading & Validation Module
Loads all sheets from the IFF Scent Duration dataset and validates integrity.
"""

import pandas as pd
import os

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProjectB_Scent_Duration_Dataset.xlsx")

# Known ingredient vocabulary (from Ingredient_Reference sheet)
TOP_NOTES = ['Bergamot', 'Lemon', 'Mandarin', 'Grapefruit', 'Orange',
             'Lavender', 'Mint', 'Basil', 'Neroli', 'Petitgrain']
MIDDLE_NOTES = ['Rose', 'Jasmine', 'Ylang-ylang', 'Geranium', 'Iris',
                'Lily', 'Tuberose', 'Violet', 'Freesia', 'Gardenia']
BASE_NOTES = ['Sandalwood', 'Cedar', 'Patchouli', 'Vetiver', 'Oakmoss',
              'Musk', 'Amber', 'Vanilla', 'Tonka Bean', 'Benzoin']
ALL_INGREDIENTS = TOP_NOTES + MIDDLE_NOTES + BASE_NOTES

INGREDIENT_NOTE_MAP = {}
for ing in TOP_NOTES:
    INGREDIENT_NOTE_MAP[ing] = 'Top'
for ing in MIDDLE_NOTES:
    INGREDIENT_NOTE_MAP[ing] = 'Middle'
for ing in BASE_NOTES:
    INGREDIENT_NOTE_MAP[ing] = 'Base'

LONGEVITY_RANGES = {
    'Short': (0, 3),
    'Medium': (3, 6),
    'Long': (6, 10),
    'Very Long': (10, float('inf'))
}

FRAGRANCE_FAMILIES = ['Chypre', 'Floral', 'Fresh', 'Gourmand', 'Oriental', 'Woody']
APPLICATION_TYPES = ['Body Lotion', 'Candle', 'Eau de Cologne', 'Eau de Parfum', 'Eau de Toilette']


def load_dataset(path=None):
    """Load the main Fragrance_Data sheet."""
    if path is None:
        path = DATASET_PATH
    df = pd.read_excel(path, sheet_name='Fragrance_Data')
    print(f"[DataLoader] Loaded {len(df)} rows, {len(df.columns)} columns from Fragrance_Data")
    return df


def load_data_dictionary(path=None):
    """Load the Data_Dictionary sheet."""
    if path is None:
        path = DATASET_PATH
    df = pd.read_excel(path, sheet_name='Data_Dictionary')
    return df


def load_summary_statistics(path=None):
    """Load the Summary_Statistics sheet."""
    if path is None:
        path = DATASET_PATH
    df = pd.read_excel(path, sheet_name='Summary_Statistics')
    return df


def load_ingredient_reference(path=None):
    """Load the Ingredient_Reference sheet and return as a DataFrame."""
    if path is None:
        path = DATASET_PATH
    df = pd.read_excel(path, sheet_name='Ingredient_Reference')
    return df


def parse_ingredients(ingredient_str):
    """Parse comma-separated ingredient string into a list of cleaned names."""
    if pd.isna(ingredient_str):
        return []
    return [ing.strip() for ing in str(ingredient_str).split(',')]


def validate_data(df):
    """Run validation checks and print a report."""
    issues = []

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values found:\n{missing[missing > 0]}")
    else:
        print("[Validation] ✓ No missing values")

    # Check for duplicate IDs
    dup_ids = df['FragranceID'].duplicated().sum()
    if dup_ids > 0:
        issues.append(f"Duplicate FragranceIDs: {dup_ids}")
    else:
        print("[Validation] ✓ No duplicate FragranceIDs")

    # Check notes sum to ~100%
    note_sums = df['TopNotes_Pct'] + df['MiddleNotes_Pct'] + df['BaseNotes_Pct']
    bad_sums = ((note_sums < 99.5) | (note_sums > 100.5)).sum()
    if bad_sums > 0:
        issues.append(f"Notes don't sum to ~100% for {bad_sums} rows")
    else:
        print("[Validation] ✓ Notes sum to ~100% for all rows")

    # Check concentration range
    conc_range = (df['Concentration_Pct'] < 0) | (df['Concentration_Pct'] > 100)
    if conc_range.sum() > 0:
        issues.append(f"Concentration out of 0-100 range for {conc_range.sum()} rows")
    else:
        print("[Validation] ✓ Concentration_Pct in valid range")

    # Check volatility score range
    vol_range = (df['VolatilityScore'] < 0) | (df['VolatilityScore'] > 1)
    if vol_range.sum() > 0:
        issues.append(f"VolatilityScore out of 0-1 range for {vol_range.sum()} rows")
    else:
        print("[Validation] ✓ VolatilityScore in valid range")

    # Check positive duration
    neg_dur = (df['ScentDuration_Hours'] <= 0).sum()
    if neg_dur > 0:
        issues.append(f"Non-positive ScentDuration for {neg_dur} rows")
    else:
        print("[Validation] ✓ All ScentDuration_Hours are positive")

    # Validate known categories
    unknown_families = set(df['FragranceFamily'].unique()) - set(FRAGRANCE_FAMILIES)
    if unknown_families:
        issues.append(f"Unknown fragrance families: {unknown_families}")
    else:
        print("[Validation] ✓ All FragranceFamily values are known")

    unknown_apps = set(df['ApplicationType'].unique()) - set(APPLICATION_TYPES)
    if unknown_apps:
        issues.append(f"Unknown application types: {unknown_apps}")
    else:
        print("[Validation] ✓ All ApplicationType values are known")

    # Validate ingredients against vocabulary
    all_parsed = df['Ingredients'].apply(parse_ingredients)
    all_found = set()
    for ing_list in all_parsed:
        all_found.update(ing_list)
    unknown_ings = all_found - set(ALL_INGREDIENTS)
    if unknown_ings:
        issues.append(f"Unknown ingredients: {unknown_ings}")
    else:
        print("[Validation] ✓ All ingredients are in the known vocabulary")

    if issues:
        print("\n[Validation] ⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[Validation] ✅ All validation checks passed!")

    return len(issues) == 0


if __name__ == '__main__':
    df = load_dataset()
    print(f"\nShape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print()
    validate_data(df)
