# ScentAI — Fragrance Intelligence Engine
## Complete Project Guide for Presentation

---

## 1. Project Overview

**Project Name:** ScentAI — Fragrance Intelligence Engine  
**Domain:** Fragrance / Perfumery + Machine Learning  
**Objective:** Predict how long a fragrance will last (in hours) based on its chemical composition, ingredient selection, and application type.

### What the Project Does (One-Liner)
> *"Given a fragrance's ingredients, concentration, note composition, and chemical properties, the system predicts the scent duration in hours using an ensemble of gradient-boosted tree models, and explains WHY the prediction is what it is using SHAP explainability."*

### Key Features
1. **Duration Prediction** — Input a fragrance recipe → get predicted hours of longevity
2. **Ingredient Optimization** — Discover which ingredient combinations maximize scent duration
3. **Model Explainability** — Understand *why* the model predicts what it does (SHAP values)
4. **Model Comparison Dashboard** — Compare ML models side-by-side
5. **Interactive Web Interface** — Beautiful React.js UI with a Flask REST API backend, featuring a Fragrance DNA Fingerprint and Scent Timeline Simulator.

---

## 2. Technology Stack

### Backend (Machine Learning + API)
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.13 | Core language |
| **pandas** | ≥2.0 | Data loading, manipulation, DataFrames |
| **numpy** | ≥1.24 | Numerical computations, array operations |
| **scikit-learn** | ≥1.3 | Train/test split, cross-validation, metrics, Random Forest |
| **XGBoost** | ≥2.0 | Gradient boosted tree model #1 |
| **LightGBM** | ≥4.0 | Gradient boosted tree model #2 (winner) |
| **Optuna** | ≥3.4 | Bayesian hyperparameter optimization |
| **SHAP** | ≥0.43 | Model explainability (SHapley Additive exPlanations) |
| **matplotlib** | ≥3.7 | EDA and SHAP plot generation |
| **seaborn** | ≥0.13 | Statistical visualizations |
| **joblib** | ≥1.3 | Model serialization (save/load) |
| **Flask** | ≥3.0 | REST API web server |
| **flask-cors** | ≥4.0 | Cross-Origin Resource Sharing for frontend |
| **openpyxl** | ≥3.1 | Reading Excel (.xlsx) datasets |

### Frontend (Web Interface)
| Technology | Purpose |
|-----------|---------|
| **React.js** (via Vite) | Component-based UI framework |
| **Vite** | Dev server and build tool |
| **Vanilla CSS** | Styling with CSS variables, glassmorphism theme, CSS Grid layout |
| **React Router** | Client-side page navigation |
| **Google Fonts** | Space Grotesk + Inter typography |

---

## 3. Dataset

**File:** `ProjectB_Scent_Duration_Dataset.xlsx`  
**Size:** 500 rows × 16 columns  
**Sheets:**
1. `Fragrance_Data` — The main data (500 fragrance compositions)
2. `Data_Dictionary` — Column descriptions
3. `Summary_Statistics` — Pre-computed stats
4. `Ingredient_Reference` — The 30 known ingredients with note types

### Column Details

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `FragranceID` | string | Unique identifier | `FRAG_001` |
| `FragranceFamily` | string (6 categories) | Classification of fragrance type | `Floral`, `Woody`, `Oriental`, `Fresh`, `Gourmand`, `Chypre` |
| `ApplicationType` | string (5 categories) | How the fragrance is applied | `Eau de Parfum`, `Eau de Toilette`, `Eau de Cologne`, `Body Lotion`, `Candle` |
| `Concentration_Pct` | float | Percentage of perfume oil in the formulation | 1.0 – 20.0 |
| `TopNotes_Pct` | float | Percentage of top notes in composition | 20.0 – 40.0 |
| `MiddleNotes_Pct` | float | Percentage of middle (heart) notes | 20.0 – 40.0 |
| `BaseNotes_Pct` | float | Percentage of base (fixative) notes | 20.0 – 60.0 |
| `Ingredients` | string | Comma-separated list of ingredient names | `"Bergamot, Rose, Sandalwood, Musk"` |
| `NumIngredients` | int | Count of ingredients | 3 – 9 |
| `AvgMolecularWeight` | int | Average molecular weight of ingredient mixture (g/mol) | 150 – 250 |
| `VolatilityScore` | float | How quickly molecules evaporate (0 = slow, 1 = fast) | 0.1 – 0.9 |
| `LogP` | float | Lipophilicity — how well molecules dissolve in oils vs water | 1.0 – 6.0 |
| `VaporPressure_Pa` | int | Vapor pressure in Pascals | 5 – 50 |
| `BoilingPoint_C` | int | Average boiling point in Celsius | 200 – 350 |
| `ScentDuration_Hours` | float | **TARGET** — How long the scent lasts | 0.5 – 24.0 |
| `LongevityCategory` | string | Duration classification | `Short` (<3h), `Medium` (3-6h), `Long` (6-10h), `Very Long` (>10h) |

### The 30 Known Ingredients (grouped by note type)

**Top Notes (10)** — Volatile, first impression, evaporate in 15-60 min:
> Bergamot, Lemon, Mandarin, Grapefruit, Orange, Lavender, Mint, Basil, Neroli, Petitgrain

**Middle Notes (10)** — Heart of the fragrance, last 2-4 hours:
> Rose, Jasmine, Ylang-ylang, Geranium, Iris, Lily, Tuberose, Violet, Freesia, Gardenia

**Base Notes (10)** — Fixatives, anchors, last 4-24+ hours:
> Sandalwood, Cedar, Patchouli, Vetiver, Oakmoss, Musk, Amber, Vanilla, Tonka Bean, Benzoin

---

## 4. Complete Project Architecture

```
Scent_prediction/
├── main.py                      # 🎯 Main pipeline orchestrator (runs everything)
├── data_loader.py               # 📦 Data loading + validation
├── data_cleaning.py             # 🧹 Cleaning + preprocessing
├── eda.py                       # 📊 Exploratory Data Analysis (8 plots)
├── feature_engineering.py       # ⚙️ Feature creation + encoding
├── model_training.py            # 🤖 Train models + Optuna tuning
├── explainability.py            # 🧠 SHAP analysis + plots
├── ingredient_optimizer.py      # 🏆 Find best ingredient combos
├── validate_predictions.py      # ✅ Validate against dataset samples
├── validate_real_world.py       # 🌍 Validate against real perfumes
├── requirements.txt             # Python dependencies
├── ProjectB_Scent_Duration_Dataset.xlsx  # Dataset
│
├── api/                         # Flask REST API
│   ├── app.py                   # Flask routes
│   └── model_service.py         # Model loading + prediction logic
│
├── frontend/                    # React.js Web UI
│   ├── index.html               # HTML entry point
│   ├── package.json             # NPM dependencies
│   ├── vite.config.js           # Vite configuration
│   └── src/
│       ├── main.jsx             # React entry point
│       ├── App.jsx              # Root component + sidebar navigation
│       ├── api.js               # API client
│       ├── index.css            # Complete CSS design system
│       ├── ingredientLookup.json # Chemical property defaults per ingredient
│       └── pages/
│           ├── PredictionPage.jsx  # Main prediction form + UI components
│           ├── OptimizerPage.jsx   # Ingredient optimization view
│           └── DashboardPage.jsx   # Model metrics dashboard
│
├── results/                     # Saved model artifacts + CSVs
│   ├── best_model.joblib        # Best trained model (LightGBM)
│   ├── all_models.joblib        # All trained models
│   ├── model_comparison.csv     # Regression metrics comparison
│   ├── classification_results.json  # Classification metrics
│   ├── feature_importance.csv   # SHAP feature importance
│   ├── ingredient_contributions.json  # Per-ingredient SHAP values
│   ├── optimal_ingredients.csv  # Top predicted combinations
│   └── top_data_combinations.csv # Best combos from actual data
│
└── plots/                       # Generated visualization images (SHAP, EDA)
```

---

## 5. Complete Pipeline Workflow

When you run `python main.py`, the following 7 phases execute in order:

### Phase 1: Data Loading (`data_loader.py`)

**Function: `load_dataset()`**
- Reads `ProjectB_Scent_Duration_Dataset.xlsx` (`Fragrance_Data` sheet)
- Validates data integrity (no nulls, duplicate IDs, correct values)

### Phase 2: Data Cleaning (`data_cleaning.py`)

**Function: `run_full_cleaning(df)`**
- Checks for duplicate rows
- Normalizes ingredients (e.g. `"bergamot"` → `"Bergamot"`)
- Detects outliers using IQR (flags but doesn't remove, as tree models handle them)
- Validates `LongevityCategory` consistency against `ScentDuration_Hours`

### Phase 3: Exploratory Data Analysis (`eda.py`)

**Function: `run_full_eda(df)`** generates 8 plots showing distribution, correlations, and feature impact.

### Phase 4: Feature Engineering (`feature_engineering.py`)

**Function: `build_feature_matrix(df)`** creates the final feature matrix:
- **Derived Features:** e.g., Base-to-Top Ratio, Volatility × VaporPressure
- **Note Counts:** Counts of Top, Middle, Base ingredients
- **Multi-Hot Encoding:** 30 binary columns (one per ingredient)
- **Categorical Encoding:** One-hot encodes Family and ApplicationType
- **Final:** ~57 features

### Phase 5: Model Training (`model_training.py`)

**Function: `run_full_training()`** trains and compares models.
- **Data Split:** 80% train / 20% test (stratified)
- **Target Transform:** `log1p(y)` due to right-skewed duration data
- **Models:** LightGBM (WINNER) and Random Forest (and previously XGBoost)
- **Hyperparameter Tuning:** Uses **Optuna** (Bayesian optimization) with 5-fold Stratified CV

**Why LightGBM Won:**
- Lowest MAE (Mean Absolute Error): ~0.576 hours
- Highest R²: ~0.9905
- Fast training with leaf-wise tree growth and histogram binning.

### Phase 6: SHAP Explainability (`explainability.py`)

**Function: `run_full_explainability()`**
- Uses `shap.TreeExplainer` to compute how much each feature contributed to the prediction.
- Generates plots like the beeswarm summary (`shap_summary.png`), waterfall charts, and dependence plots to show the nonlinear impact of features (e.g., how higher concentration pushes duration up).

### Phase 7: Ingredient Optimization (`ingredient_optimizer.py`)

**Function: `run_full_optimization()`**
- Analyzes actual data for top combos.
- Uses SHAP to find the most influential individual ingredients.
- Simulates thousands of 4-6 ingredient combinations and predicts their duration to find the theoretical best recipes.

---

## 6. Flask REST API (`api/`)

### Model Service (`model_service.py`)
- Loads the LightGBM model and required metadata on startup.
- **`predict(input_data)`**:
  1. Applies full feature engineering pipeline to the input JSON.
  2. Runs `model.predict()` and reverses the log-transform to get hours.
  3. Uses SHAP to extract the top positive (increasing) and negative (decreasing) factors for this specific prediction.
  4. Returns prediction details and explainability metrics.

### Flask Routes (`app.py`)
- `/api/predict`: POST endpoint for predictions
- `/api/ingredients`, `/api/options`: GET endpoints to supply the UI
- `/api/model-info`, `/api/optimize`: GET endpoints for dashboards

---

## 7. React.js Frontend (`frontend/`)

### Design Theme
- **"Dark Glassmorphism + Bioluminescent Perfume Lab"**
- Two-column responsive layout (55/45 split on desktop, collapses to single column on mobile).
- Colors: Deep space black (#050508) background, frosted glass cards `rgba(255,255,255,0.04)` with blur, neon teal/violet/rose gold accents.

### Page 1: Predict Duration (`PredictionPage.jsx`)

**Left column (Form):**
- **Fragrance Profile:** Family, Application Type, Concentration slider.
- **Note Composition:** Top/Middle/Base sliders (adds up to 100%).
- **Select Ingredients:** 30 chips. Selection auto-fills average chemistry using `ingredientLookup.json`.
- **Chemical Properties:** Collapsible, auto-filled or overridden manually.

**Right column (Results & Visualization):**
- **Result Card:** Glowing duration number, SVG arc gauge speedometer, Longevity category, and SHAP explainability pills (Green = increases duration, Red = decreases).
- **Fragrance DNA Fingerprint (NEW):** An animated hexagonal radar/spider chart mapping the fragrance's identity across 6 axes (Longevity, Freshness, Depth, Warmth, Complexity, Sillage).
- **Scent Timeline Simulator (NEW):** An interactive SVG timeline simulating how the fragrance evolves. Features a draggable scrubber and dynamically highlights which ingredients (Top/Middle/Base) are active at a given hour.
- **Pro Tips:** Dynamic suggestions based on the prediction.

### Page 2: Ingredient Optimizer (`OptimizerPage.jsx`)
- Top 25 predicted combinations with gold/silver/bronze highlights.
- Animated horizontal bars showing SHAP impact per ingredient.

### Page 3: Model Dashboard (`DashboardPage.jsx`)
- Compares models visually (MAE, R²).
- Explains model metrics (RMSE, MAPE) and global feature importance.

---

## 8. Validation Results

**Internal Validation:** On held-out datasets, LightGBM achieves R² > 0.99 with < 40 mins error.
**Real-World Validation (`validate_real_world.py`):**
When tested against real perfumes (like Dior Sauvage, Chanel No. 5), predictions are off. Why?
1. The model learned perfectly on a *synthetic/curated* dataset, not actual lab chemistry.
2. The model only knows 30 ingredients, while real perfumes have 50-200+ (missing Iso E Super, Ambroxan).
3. The UI auto-fills *average* chemical properties for ingredients, which might not reflect actual molecular formulations.

---

## 9. Key Concepts for Presentation Q&A

**"What is Optuna doing?"**
It uses Bayesian optimization to test different hyperparameters intelligently, learning which combinations lead to better models rather than guessing randomly.

**"What is SHAP?"**
Based on game theory, it breaks down a prediction to show exactly how much each feature contributed. E.g., Base notes +1.5h, Volatility -0.5h.

**"Why log-transform the target?"**
The scent duration target is highly right-skewed. `log1p` transformation normalizes it, allowing gradient-boosted models to perform much better.

---

## 10. How to Run the Project

1. **Train Model:** `python main.py` (Takes 5-10 mins for hyperparameter tuning)
2. **Start Backend:** `python api/app.py` (Runs on port 5000)
3. **Start Frontend:** `cd frontend && npm install && npm run dev` (Runs on port 5173)
4. Open `http://localhost:5173` in a browser.
