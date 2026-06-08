from __future__ import annotations

import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Scent Duration Predictor", page_icon="IFF", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.pkl"
CLASSIFIER_PATH = ROOT / "classification_model.pkl"
METADATA_PATH = ROOT / "model_metadata.pkl"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error(
            "Model artifacts were not found. Run scent_duration_prediction.ipynb first "
            "to generate best_model.pkl, classification_model.pkl, and model_metadata.pkl."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    classifier = joblib.load(CLASSIFIER_PATH) if CLASSIFIER_PATH.exists() else None
    return model, classifier, metadata


def concentration_tier(concentration):
    if concentration < 5:
        return "EdC"
    if concentration <= 15:
        return "EdT"
    if concentration <= 20:
        return "EdP"
    return "Perfume"


def encode_label(metadata, column, value):
    classes = metadata["label_classes"].get(column, [])
    value = str(value)
    if value in classes:
        return classes.index(value)
    return 0


def normalize_ingredient(value):
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def category_from_duration(hours):
    if hours < 4:
        return "Short"
    if hours < 8:
        return "Medium"
    if hours < 12:
        return "Long"
    return "Very Long"


def build_feature_row(
    metadata,
    fragrance_family,
    application_type,
    concentration_pct,
    top_notes_pct,
    middle_notes_pct,
    base_notes_pct,
    ingredients,
    avg_molecular_weight,
    volatility_score,
    logp,
    vapor_pressure_pa,
    boiling_point_c,
):
    row = dict(metadata["training_defaults"])
    feature_cols = metadata["feature_cols"]
    key_fixatives = metadata["key_fixatives"]
    ingredient_feature_names = metadata["ingredient_feature_names"]

    ingredient_values = [normalize_ingredient(item) for item in ingredients]
    ingredient_text = " | ".join(ingredient_values)

    row.update(
        {
            "Concentration_Pct": concentration_pct,
            "TopNotes_Pct": top_notes_pct,
            "MiddleNotes_Pct": middle_notes_pct,
            "BaseNotes_Pct": base_notes_pct,
            "NumIngredients": len(ingredient_values),
            "AvgMolecularWeight": avg_molecular_weight,
            "VolatilityScore": volatility_score,
            "LogP": logp,
            "VaporPressure_Pa": vapor_pressure_pa,
            "BoilingPoint_C": boiling_point_c,
            "FragranceFamily_encoded": encode_label(metadata, "FragranceFamily", fragrance_family),
            "ApplicationType_encoded": encode_label(metadata, "ApplicationType", application_type),
            "concentration_tier_encoded": encode_label(metadata, "concentration_tier", concentration_tier(concentration_pct)),
        }
    )

    row["base_to_top_ratio"] = row["BaseNotes_Pct"] / (row["TopNotes_Pct"] + 1)
    row["fixative_score"] = row["BaseNotes_Pct"] * row["LogP"]
    row["volatility_adjusted_concentration"] = row["Concentration_Pct"] / (row["VolatilityScore"] + 0.01)
    row["ingredient_complexity"] = row["NumIngredients"] * row["AvgMolecularWeight"]
    row["boiling_vapor_ratio"] = row["BoilingPoint_C"] / (row["VaporPressure_Pa"] + 1)

    for fixative in key_fixatives:
        col = f"has_{fixative.replace(' ', '_')}"
        row[col] = int(fixative in ingredient_text)
    row["fixative_count"] = sum(row[f"has_{fixative.replace(' ', '_')}"] for fixative in key_fixatives)

    selected_set = set(ingredient_values)
    for feature in ingredient_feature_names:
        ingredient_name = feature.replace("ingredient__", "", 1)
        row[feature] = int(ingredient_name in selected_set)

    return pd.DataFrame([row], columns=feature_cols).replace([np.inf, -np.inf], np.nan)


def gauge_chart(hours):
    capped_hours = max(0, min(24, float(hours)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=capped_hours,
            number={"suffix": " h"},
            gauge={
                "axis": {"range": [0, 24], "tickwidth": 1},
                "bar": {"color": "#287271"},
                "steps": [
                    {"range": [0, 4], "color": "#FDE2E2"},
                    {"range": [4, 8], "color": "#FFF3BF"},
                    {"range": [8, 12], "color": "#D8F3DC"},
                    {"range": [12, 24], "color": "#B7E4C7"},
                ],
                "threshold": {"line": {"color": "#D62828", "width": 4}, "thickness": 0.75, "value": capped_hours},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def show_shap_waterfall(model, feature_row):
    try:
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.Explainer(model.predict, feature_row)
        shap_values = explainer(feature_row)
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    except Exception as exc:
        st.info(f"SHAP waterfall is unavailable for this prediction: {exc}")


model, classifier, metadata = load_artifacts()

st.title("Scent Duration Prediction")
st.caption("Predict fragrance longevity from formulation structure, application format, and ingredient choices.")

left, right = st.columns([0.95, 1.05])

with left:
    st.subheader("Formula Inputs")
    family_options = metadata["label_classes"].get("FragranceFamily", [])
    application_options = metadata["label_classes"].get("ApplicationType", [])
    ingredient_options = metadata.get("ingredient_options", [])
    defaults = metadata["training_defaults"]

    fragrance_family = st.selectbox("Fragrance family", family_options)
    application_type = st.selectbox("Application type", application_options)

    concentration_pct = st.slider("Concentration (%)", 0.0, 35.0, float(defaults.get("Concentration_Pct", 15.0)), 0.5)
    top_notes_pct = st.slider("Top notes (%)", 0.0, 100.0, float(defaults.get("TopNotes_Pct", 30.0)), 1.0)
    middle_notes_pct = st.slider("Middle notes (%)", 0.0, 100.0, float(defaults.get("MiddleNotes_Pct", 40.0)), 1.0)
    base_notes_pct = st.slider("Base notes (%)", 0.0, 100.0, float(defaults.get("BaseNotes_Pct", 30.0)), 1.0)

    total_notes = top_notes_pct + middle_notes_pct + base_notes_pct
    if not math.isclose(total_notes, 100.0, abs_tol=1.0):
        st.warning(f"Top + middle + base notes sum to {total_notes:.1f}%. A total near 100% is recommended.")

    ingredients = st.multiselect("Ingredients", ingredient_options)

    with st.expander("Advanced physicochemical inputs", expanded=False):
        avg_molecular_weight = st.number_input(
            "Average molecular weight (g/mol)",
            min_value=0.0,
            value=float(defaults.get("AvgMolecularWeight", 180.0)),
            step=1.0,
        )
        volatility_score = st.slider(
            "Volatility score",
            0.0,
            1.0,
            float(defaults.get("VolatilityScore", 0.45)),
            0.01,
        )
        logp = st.number_input("LogP", value=float(defaults.get("LogP", 3.0)), step=0.1)
        vapor_pressure_pa = st.number_input(
            "Vapor pressure (Pa)",
            min_value=0.0,
            value=float(defaults.get("VaporPressure_Pa", 10.0)),
            step=0.1,
        )
        boiling_point_c = st.number_input(
            "Boiling point (C)",
            min_value=0.0,
            value=float(defaults.get("BoilingPoint_C", 250.0)),
            step=1.0,
        )

    predict_clicked = st.button("Predict Duration", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")
    if predict_clicked:
        feature_row = build_feature_row(
            metadata,
            fragrance_family,
            application_type,
            concentration_pct,
            top_notes_pct,
            middle_notes_pct,
            base_notes_pct,
            ingredients,
            avg_molecular_weight,
            volatility_score,
            logp,
            vapor_pressure_pa,
            boiling_point_c,
        )

        predicted_hours = float(model.predict(feature_row)[0])
        if classifier is not None:
            class_index = int(classifier.predict(feature_row)[0])
            class_labels = metadata.get("class_labels", [])
            predicted_category = class_labels[class_index] if class_index < len(class_labels) else category_from_duration(predicted_hours)
        else:
            predicted_category = category_from_duration(predicted_hours)

        metric_a, metric_b = st.columns(2)
        metric_a.metric("Predicted duration", f"{predicted_hours:.2f} hours")
        metric_b.metric("Longevity category", predicted_category)
        st.plotly_chart(gauge_chart(predicted_hours), use_container_width=True)

        st.subheader("Prediction Explanation")
        show_shap_waterfall(model, feature_row)

        with st.expander("Model input row"):
            st.dataframe(feature_row.T.rename(columns={0: "value"}), use_container_width=True)
    else:
        st.info("Set the formula inputs and click Predict Duration.")
