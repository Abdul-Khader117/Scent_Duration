# ScentAI — Fragrance Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646cff.svg)](https://vitejs.dev/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20RandomForest-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)

An end-to-end Machine Learning intelligence engine that predicts fragrance longevity (in hours) and longevity categories from chemical composition, olfactory pyramid note ratios, and ingredient formulations. Features SHAP explainability, combinatorial ingredient optimization, and a modern React web interface.

---

## 🌟 Key Features

1. **Duration & Category Prediction**:
   - Predicts exact scent duration ($R^2 > 0.99$, $\text{MAE} \approx 0.58\text{ hours}$) based on concentration, olfactory notes, and physicochemical properties (vapor pressure, boiling point, molecular weight, LogP, volatility score).
   - Classifies longevity into *Short* (<3h), *Medium* (3–6h), *Long* (6–10h), and *Very Long* (>10h).

2. **SHAP Local & Global Explainability**:
   - Explains *why* a specific composition lasts with positive and negative feature impact attribution using Tree SHAP.

3. **Combinatorial Ingredient Optimizer**:
   - Simulates thousands of ingredient permutations to find optimal top, middle, and base note blends for maximum longevity.

4. **Model Performance Dashboard**:
   - Compare regression and classification metrics across gradient-boosted trees and ensemble models with confusion matrices and residual analysis.

5. **Modern Interactive Web Interface**:
   - Sleek React 19 UI with real-time prediction simulator, interactive sliders, dynamic ingredient pickers, and live radar and timeline visualizations.

---

## 🏗️ Project Architecture

```
Scent_Duration/
├── backend/
│   ├── api/
│   │   ├── app.py                     # Flask REST API server
│   │   └── model_service.py           # Model inference & SHAP explainability service
│   ├── data_loader.py                 # Dataset loader & domain constants
│   ├── data_cleaning.py               # Outlier detection, cleaning & validation
│   ├── eda.py                         # Exploratory Data Analysis & visual plots
│   ├── feature_engineering.py         # Chemical interactions & multi-hot encoding
│   ├── model_training.py              # XGBoost, RF tuning with Optuna & cross-validation
│   ├── explainability.py              # SHAP summary & bar plot generators
│   ├── ingredient_optimizer.py        # Top ingredient combination search engine
│   ├── validate_predictions.py        # Prediction validation checks
│   ├── validate_real_world.py         # Real-world benchmark test cases
│   ├── main.py                        # Full pipeline orchestrator
│   ├── requirements.txt               # Backend Python dependencies
│   ├── plots/                         # Generated visualizations
│   └── results/                       # Trained model artifacts (.joblib) & metrics (.csv, .json)
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── PredictionPage.jsx     # Fragrance builder & duration predictor
│   │   │   ├── OptimizerPage.jsx      # Top ingredient blend optimizer
│   │   │   └── DashboardPage.jsx      # Model metrics & evaluation dashboard
│   │   ├── api.js                     # Backend API client
│   │   ├── App.jsx                    # Root app layout & routing
│   │   └── index.css                  # Modern UI design system
│   ├── package.json
│   └── vite.config.js
├── Model comparison/                  # Comparison assets & benchmarks
├── project complete.md                # In-depth project & presentation guide
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 2. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Run the complete ML pipeline to retrain models
python main.py

# Start the Flask REST API server (runs on http://localhost:5000)
python api/app.py
```

### 3. Frontend Setup

```bash
# In a new terminal, navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite dev server (runs on http://localhost:5173)
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 📊 Model Performance

| Model | MAE (hours) | RMSE (hours) | $R^2$ | MAPE (%) |
|---|---|---|---|---|
| **XGBoost** | **0.58 h** | **0.80 h** | **0.991** | **9.87%** |
| **Random Forest** | 0.93 h | 1.36 h | 0.974 | 14.40% |

---

## 🛠️ API Endpoints

- `GET /api/health` — Health check and model readiness status
- `POST /api/predict` — Predict scent duration with local SHAP feature breakdown
- `GET /api/ingredients` — Retrieve available top, middle, and base notes
- `GET /api/options` — Retrieve fragrance families and application types
- `GET /api/model-info` — Performance comparison metrics
- `GET /api/optimize?limit=15` — Top longevity ingredient combinations

---

## 📄 License

MIT License. Designed and developed for fragrance longevity intelligence.
