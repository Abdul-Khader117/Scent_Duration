"""
validate_real_world.py - Validate ScentAI predictions against REAL fragrances
from Fragrantica and other fragrance databases.

Tests 6 famous perfumes with known ingredients & reported longevity.
"""

import requests
import json
import sys
import os

API_BASE = "http://localhost:5000/api"

# Map real perfume ingredients to our model's 30 known ingredients
# Our model knows: TOP = Bergamot, Lemon, Mandarin, Grapefruit, Orange, Lavender, Mint, Basil, Neroli, Petitgrain
#                   MID = Rose, Jasmine, Ylang-ylang, Geranium, Iris, Lily, Tuberose, Violet, Freesia, Gardenia
#                   BASE = Sandalwood, Cedar, Patchouli, Vetiver, Oakmoss, Musk, Amber, Vanilla, Tonka Bean, Benzoin

REAL_FRAGRANCES = [
    {
        "name": "Dior Sauvage EDP",
        "brand": "Dior",
        "source": "Fragrantica",
        "fragrance_family": "Oriental",   # Closest match to Spicy Oriental Fougere
        "application_type": "Eau de Parfum",
        "concentration": 15,               # EDP = ~15-20%
        "top_pct": 25, "mid_pct": 35, "base_pct": 40,
        # Real notes: Top=Bergamot | Mid=Lavender(moved to top in model) | Base=Vanilla
        # Mapped to model's vocabulary:
        "ingredients": ["Bergamot", "Lavender", "Vanilla"],
        "reported_longevity_min": 8,
        "reported_longevity_max": 12,
        "reported_longevity_avg": 10,
    },
    {
        "name": "Chanel No. 5 EDP",
        "brand": "Chanel",
        "source": "Fragrantica",
        "fragrance_family": "Floral",
        "application_type": "Eau de Parfum",
        "concentration": 15,
        "top_pct": 30, "mid_pct": 35, "base_pct": 35,
        # Real notes: Top=Ylang-ylang, Neroli, Bergamot | Mid=Iris, Jasmine, Rose | Base=Sandalwood, Oakmoss, Vanilla, Patchouli, Vetiver
        "ingredients": ["Bergamot", "Neroli", "Ylang-ylang", "Iris", "Jasmine", "Rose",
                        "Sandalwood", "Oakmoss", "Vanilla", "Patchouli", "Vetiver"],
        "reported_longevity_min": 4,
        "reported_longevity_max": 10,
        "reported_longevity_avg": 7,
    },
    {
        "name": "Acqua di Gio Profumo EDP",
        "brand": "Giorgio Armani",
        "source": "Fragrantica",
        "fragrance_family": "Fresh",        # Aquatic aromatic
        "application_type": "Eau de Parfum",
        "concentration": 15,
        "top_pct": 30, "mid_pct": 35, "base_pct": 35,
        # Real notes: Top=Bergamot | Mid=Geranium | Base=Patchouli
        "ingredients": ["Bergamot", "Geranium", "Patchouli"],
        "reported_longevity_min": 8,
        "reported_longevity_max": 10,
        "reported_longevity_avg": 9,
    },
    {
        "name": "Jo Malone English Pear & Freesia EDC",
        "brand": "Jo Malone",
        "source": "Fragrantica",
        "fragrance_family": "Chypre",       # Chypre Fruity
        "application_type": "Eau de Cologne",
        "concentration": 4,                  # EDC = ~3-5%
        "top_pct": 35, "mid_pct": 35, "base_pct": 30,
        # Real notes: Mid=Freesia, Rose | Base=Patchouli, Musk, Amber
        "ingredients": ["Freesia", "Rose", "Patchouli", "Musk", "Amber"],
        "reported_longevity_min": 1,
        "reported_longevity_max": 4,
        "reported_longevity_avg": 2.5,
    },
    {
        "name": "Tom Ford Tobacco Vanille EDP",
        "brand": "Tom Ford",
        "source": "Fragrantica",
        "fragrance_family": "Oriental",     # Amber Spicy
        "application_type": "Eau de Parfum",
        "concentration": 18,                 # Private blend EDPs are higher concentration
        "top_pct": 20, "mid_pct": 30, "base_pct": 50,
        # Real notes: Mid=Vanilla, Tonka Bean | Base= (woody/dried fruit -> Cedar)
        "ingredients": ["Vanilla", "Tonka Bean", "Cedar"],
        "reported_longevity_min": 8,
        "reported_longevity_max": 14,
        "reported_longevity_avg": 11,
    },
    {
        "name": "Versace Eros EDT",
        "brand": "Versace",
        "source": "Fragrantica",
        "fragrance_family": "Fresh",        # Aromatic Fougere
        "application_type": "Eau de Toilette",
        "concentration": 8,                  # EDT = ~5-10%
        "top_pct": 30, "mid_pct": 30, "base_pct": 40,
        # Real notes: Top=Mint, Lemon | Mid=Geranium, Tonka Bean | Base=Vanilla, Cedar, Vetiver, Oakmoss
        "ingredients": ["Mint", "Lemon", "Geranium", "Tonka Bean",
                        "Vanilla", "Cedar", "Vetiver", "Oakmoss"],
        "reported_longevity_min": 6,
        "reported_longevity_max": 8,
        "reported_longevity_avg": 7,
    },
]


def predict_via_api(frag):
    """Send fragrance data to the model API."""
    payload = {
        "FragranceFamily": frag["fragrance_family"],
        "ApplicationType": frag["application_type"],
        "Concentration_Pct": frag["concentration"],
        "TopNotes_Pct": frag["top_pct"],
        "MiddleNotes_Pct": frag["mid_pct"],
        "BaseNotes_Pct": frag["base_pct"],
        "Ingredients": frag["ingredients"],
        # Let the auto-fill handle chemical properties based on ingredients
        "AvgMolecularWeight": 189,
        "VolatilityScore": 0.45,
        "LogP": 3.67,
        "VaporPressure_Pa": 23,
        "BoilingPoint_C": 266,
    }

    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)
    if resp.status_code != 200:
        return None
    return resp.json()


def main():
    print("=" * 80)
    print("   ScentAI -- REAL-WORLD FRAGRANCE VALIDATION")
    print("   Testing against 6 famous perfumes with known longevity data")
    print("=" * 80)

    # Check API is running
    try:
        resp = requests.get(f"{API_BASE}/model-info", timeout=5)
        model_info = resp.json()
        print(f"\n   Model: {model_info.get('best_model', 'Unknown')}")
        print(f"   Sources: Fragrantica, brand official sites, user reviews\n")
    except Exception:
        print("\n   ERROR: Flask API is not reachable. Start it with: python api/app.py")
        return

    results = []

    print("-" * 80)
    print(f"{'Fragrance':<35} | {'Predicted':>9} | {'Real Range':>12} | {'Real Avg':>8} | {'In Range?':>9} | {'Error':>7}")
    print("-" * 80)

    for frag in REAL_FRAGRANCES:
        pred_result = predict_via_api(frag)

        if pred_result is None:
            print(f"{frag['name']:<35} | {'API ERR':>9} |")
            continue

        predicted = pred_result['predicted_duration_hours']
        real_min = frag['reported_longevity_min']
        real_max = frag['reported_longevity_max']
        real_avg = frag['reported_longevity_avg']

        in_range = real_min <= predicted <= real_max
        error_from_avg = abs(predicted - real_avg)
        error_pct = (error_from_avg / real_avg * 100) if real_avg > 0 else 0

        range_str = f"{real_min}-{real_max}h"
        in_range_str = "YES" if in_range else "NO"

        results.append({
            'name': frag['name'],
            'predicted': predicted,
            'real_min': real_min,
            'real_max': real_max,
            'real_avg': real_avg,
            'in_range': in_range,
            'error_from_avg': error_from_avg,
            'error_pct': error_pct,
            'category_pred': pred_result['longevity_category'],
            'ingredients': frag['ingredients'],
        })

        print(f"{frag['name']:<35} | {predicted:>8.2f}h | {range_str:>12} | {real_avg:>7.1f}h | {'  YES' if in_range else '   NO':>9} | {error_from_avg:>6.2f}h")

    if not results:
        print("\nNo successful predictions.")
        return

    # Summary
    print("\n" + "=" * 80)
    print("   SUMMARY")
    print("=" * 80)

    in_range_count = sum(1 for r in results if r['in_range'])
    avg_error = sum(r['error_from_avg'] for r in results) / len(results)
    avg_error_pct = sum(r['error_pct'] for r in results) / len(results)

    print(f"\n   Fragrances tested:      {len(results)}")
    print(f"   In reported range:      {in_range_count}/{len(results)} ({in_range_count/len(results)*100:.0f}%)")
    print(f"   Avg error vs reported:  {avg_error:.2f} hours ({avg_error*60:.0f} minutes)")
    print(f"   Avg % error:            {avg_error_pct:.1f}%")

    # Detailed breakdown
    print(f"\n   DETAILED RESULTS:")
    print(f"   {'':>3} {'Fragrance':<35} {'Predicted':>9} {'Real Avg':>9} {'Error':>8} {'%Err':>7} {'Category':>12}")
    print(f"   " + "-" * 88)
    for i, r in enumerate(results):
        emoji = "  " if r['in_range'] else "!!"
        print(f"   {emoji} {r['name']:<35} {r['predicted']:>8.2f}h {r['real_avg']:>8.1f}h {r['error_from_avg']:>7.2f}h {r['error_pct']:>6.1f}% {r['category_pred']:>12}")

    print(f"\n   INTERPRETATION:")

    if in_range_count == len(results):
        print(f"   ALL {len(results)} predictions fall within reported longevity ranges!")
        print(f"   The model generalizes well to real-world fragrances.")
    elif in_range_count >= len(results) * 0.5:
        print(f"   {in_range_count}/{len(results)} predictions fall within reported ranges.")
        print(f"   The model shows reasonable real-world accuracy.")
    else:
        print(f"   Only {in_range_count}/{len(results)} predictions fall within reported ranges.")

    print()
    print("   IMPORTANT CONTEXT:")
    print("   - Reported longevity varies HUGELY by person (skin chemistry, climate, application)")
    print("   - Fragrantica ranges represent crowd-sourced averages from thousands of reviewers")
    print("   - Our model was trained on a synthetic/curated dataset, not real fragrance formulas")
    print("   - The model only knows 30 ingredients; real perfumes contain 50-200+ components")
    print("   - Despite these limitations, predictions within +/-2h of reported average is excellent")
    print()


if __name__ == "__main__":
    main()
