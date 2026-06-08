# Scent Duration Prediction System

End-to-end Python ML project for predicting fragrance scent duration and longevity category from formula structure, ingredients, and physicochemical properties.

## Files

- `scent_duration_prediction.ipynb` - complete notebook for EDA, feature engineering, modeling, tuning, SHAP, ingredient analysis, and report export.
- `app.py` - Streamlit prediction app using the saved model artifacts.
- `requirements.txt` - Python dependencies.

## How to run

1. Place `ProjectB_Scent_Duration_Dataset.xlsx` in this folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run all cells in `scent_duration_prediction.ipynb`.
4. Start the Streamlit app:

```bash
streamlit run app.py
```

The notebook saves `best_model.pkl`, `classification_model.pkl`, `model_metadata.pkl`, model performance tables, figures, and `reports/scent_duration_report.html`.
