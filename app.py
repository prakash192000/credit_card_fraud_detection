<<<<<<< HEAD
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
BEST_MODEL_FILE = MODEL_DIR / "best_model.joblib"
BEST_DL_MODEL_FILE = MODEL_DIR / "best_deep_learning_model.keras"
METADATA_FILE = MODEL_DIR / "model_metadata.joblib"
MODEL_COMPARISON_FILE = MODEL_DIR / "model_comparison.csv"


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


@st.cache_resource
def load_artifacts():
    if not METADATA_FILE.exists():
        st.error("Model metadata not found. Run the Jupyter notebook first.")
        st.stop()

    metadata = joblib.load(METADATA_FILE)
    scaler = metadata["scaler"]
    feature_names = metadata["feature_names"]
    model_type = metadata["best_model_type"]

    if model_type == "deep_learning":
        try:
            import tensorflow as tf
        except ImportError:
            st.error("TensorFlow is required to load the deep learning model.")
            st.stop()
        model = tf.keras.models.load_model(BEST_DL_MODEL_FILE)
    else:
        model = joblib.load(BEST_MODEL_FILE)

    return model, scaler, feature_names, metadata


@st.cache_data
def load_model_comparison():
    if not MODEL_COMPARISON_FILE.exists():
        return pd.DataFrame()

    comparison_df = pd.read_csv(MODEL_COMPARISON_FILE)
    return comparison_df.sort_values("f1_score", ascending=False).reset_index(drop=True)


def predict(model, scaler, feature_names, input_df, model_type):
    ordered_df = input_df[feature_names].copy()
    scaled_values = scaler.transform(ordered_df)

    if model_type == "deep_learning":
        probabilities = model.predict(scaled_values).ravel()
    else:
        probabilities = model.predict_proba(scaled_values)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)
    result_df = input_df.copy()
    result_df["Fraud Probability"] = probabilities
    result_df["Prediction"] = np.where(predictions == 1, "Fraud", "Genuine")
    return result_df


st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")
st.caption("Machine Learning and Deep Learning based fraud transaction prediction")

model, scaler, feature_names, metadata = load_artifacts()
comparison_df = load_model_comparison()

overview_cols = st.columns(3)
overview_cols[0].metric("Best Model", metadata["best_model_name"])
overview_cols[1].metric("Best Model Type", metadata["best_model_type"].replace("_", " ").title())
overview_cols[2].metric("Models Tried", len(comparison_df) if not comparison_df.empty else "N/A")

st.divider()

tab_models, tab_batch, tab_manual = st.tabs(["Model Comparison", "Batch Prediction", "Manual Prediction"])

with tab_models:
    st.subheader("Models Tried")

    if comparison_df.empty:
        st.warning("Model comparison file not found. Run the Jupyter notebook to generate model metrics.")
    else:
        best_model_name = metadata["best_model_name"]
        display_df = comparison_df.copy()
        display_df["selected"] = np.where(display_df["model_name"] == best_model_name, "Best Model", "")

        model_cards = st.columns(len(display_df))
        for column, row in zip(model_cards, display_df.itertuples(index=False)):
            with column:
                st.metric(row.model_name, f"{row.f1_score:.3f}", help="F1 score")
                st.caption(row.model_type.replace("_", " ").title())
                if row.model_name == best_model_name:
                    st.success("Selected best model")

        st.markdown("#### Complete Metrics")
        formatted_df = display_df.rename(
            columns={
                "model_name": "Model",
                "model_type": "Type",
                "selected": "Selection",
                **METRIC_LABELS,
            }
        )
        formatted_df["Type"] = formatted_df["Type"].str.replace("_", " ").str.title()
        for metric_label in METRIC_LABELS.values():
            formatted_df[metric_label] = (formatted_df[metric_label] * 100).round(2)

        st.dataframe(
            formatted_df[
                ["Selection", "Model", "Type", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Metric Charts")
        chart_metric = st.selectbox(
            "Select metric",
            options=METRIC_COLUMNS,
            format_func=lambda metric: METRIC_LABELS[metric],
            index=METRIC_COLUMNS.index("f1_score"),
        )
        chart_df = display_df.set_index("model_name")[[chart_metric]].rename(columns={chart_metric: METRIC_LABELS[chart_metric]})
        st.bar_chart(chart_df, use_container_width=True)

        st.caption(
            "For imbalanced fraud detection, F1 score, recall, precision, and PR-AUC are usually more useful than accuracy alone."
        )

with tab_batch:
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        missing_features = [feature for feature in feature_names if feature not in input_data.columns]

        if missing_features:
            st.error(f"Missing required columns: {', '.join(missing_features)}")
        else:
            predictions = predict(
                model,
                scaler,
                feature_names,
                input_data,
                metadata["best_model_type"],
            )
            st.dataframe(predictions, use_container_width=True)
            st.download_button(
                "Download Predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

with tab_manual:
    st.write("Enter one transaction's values. The Kaggle dataset uses PCA features V1 to V28.")
    values = {}

    cols = st.columns(3)
    for index, feature in enumerate(feature_names):
        default_value = 0.0
        if feature == "Amount":
            default_value = 100.0
        values[feature] = cols[index % 3].number_input(feature, value=default_value, format="%.6f")

    if st.button("Predict Transaction"):
        manual_df = pd.DataFrame([values])
        prediction = predict(model, scaler, feature_names, manual_df, metadata["best_model_type"])
        probability = prediction.loc[0, "Fraud Probability"]
        label = prediction.loc[0, "Prediction"]

        if label == "Fraud":
            st.error(f"Prediction: Fraud transaction. Probability: {probability:.4f}")
        else:
            st.success(f"Prediction: Genuine transaction. Probability: {probability:.4f}")
=======
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
BEST_MODEL_FILE = MODEL_DIR / "best_model.joblib"
BEST_DL_MODEL_FILE = MODEL_DIR / "best_deep_learning_model.keras"
METADATA_FILE = MODEL_DIR / "model_metadata.joblib"
MODEL_COMPARISON_FILE = MODEL_DIR / "model_comparison.csv"


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


@st.cache_resource
def load_artifacts():
    if not METADATA_FILE.exists():
        st.error("Model metadata not found. Run the Jupyter notebook first.")
        st.stop()

    metadata = joblib.load(METADATA_FILE)
    scaler = metadata["scaler"]
    feature_names = metadata["feature_names"]
    model_type = metadata["best_model_type"]

    if model_type == "deep_learning":
        try:
            import tensorflow as tf
        except ImportError:
            st.error("TensorFlow is required to load the deep learning model.")
            st.stop()
        model = tf.keras.models.load_model(BEST_DL_MODEL_FILE)
    else:
        model = joblib.load(BEST_MODEL_FILE)

    return model, scaler, feature_names, metadata


@st.cache_data
def load_model_comparison():
    if not MODEL_COMPARISON_FILE.exists():
        return pd.DataFrame()

    comparison_df = pd.read_csv(MODEL_COMPARISON_FILE)
    return comparison_df.sort_values("f1_score", ascending=False).reset_index(drop=True)


def predict(model, scaler, feature_names, input_df, model_type):
    ordered_df = input_df[feature_names].copy()
    scaled_values = scaler.transform(ordered_df)

    if model_type == "deep_learning":
        probabilities = model.predict(scaled_values).ravel()
    else:
        probabilities = model.predict_proba(scaled_values)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)
    result_df = input_df.copy()
    result_df["Fraud Probability"] = probabilities
    result_df["Prediction"] = np.where(predictions == 1, "Fraud", "Genuine")
    return result_df


st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")
st.caption("Machine Learning and Deep Learning based fraud transaction prediction")

model, scaler, feature_names, metadata = load_artifacts()
comparison_df = load_model_comparison()

overview_cols = st.columns(3)
overview_cols[0].metric("Best Model", metadata["best_model_name"])
overview_cols[1].metric("Best Model Type", metadata["best_model_type"].replace("_", " ").title())
overview_cols[2].metric("Models Tried", len(comparison_df) if not comparison_df.empty else "N/A")

st.divider()

tab_models, tab_batch, tab_manual = st.tabs(["Model Comparison", "Batch Prediction", "Manual Prediction"])

with tab_models:
    st.subheader("Models Tried")

    if comparison_df.empty:
        st.warning("Model comparison file not found. Run the Jupyter notebook to generate model metrics.")
    else:
        best_model_name = metadata["best_model_name"]
        display_df = comparison_df.copy()
        display_df["selected"] = np.where(display_df["model_name"] == best_model_name, "Best Model", "")

        model_cards = st.columns(len(display_df))
        for column, row in zip(model_cards, display_df.itertuples(index=False)):
            with column:
                st.metric(row.model_name, f"{row.f1_score:.3f}", help="F1 score")
                st.caption(row.model_type.replace("_", " ").title())
                if row.model_name == best_model_name:
                    st.success("Selected best model")

        st.markdown("#### Complete Metrics")
        formatted_df = display_df.rename(
            columns={
                "model_name": "Model",
                "model_type": "Type",
                "selected": "Selection",
                **METRIC_LABELS,
            }
        )
        formatted_df["Type"] = formatted_df["Type"].str.replace("_", " ").str.title()
        for metric_label in METRIC_LABELS.values():
            formatted_df[metric_label] = (formatted_df[metric_label] * 100).round(2)

        st.dataframe(
            formatted_df[
                ["Selection", "Model", "Type", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Metric Charts")
        chart_metric = st.selectbox(
            "Select metric",
            options=METRIC_COLUMNS,
            format_func=lambda metric: METRIC_LABELS[metric],
            index=METRIC_COLUMNS.index("f1_score"),
        )
        chart_df = display_df.set_index("model_name")[[chart_metric]].rename(columns={chart_metric: METRIC_LABELS[chart_metric]})
        st.bar_chart(chart_df, use_container_width=True)

        st.caption(
            "For imbalanced fraud detection, F1 score, recall, precision, and PR-AUC are usually more useful than accuracy alone."
        )

with tab_batch:
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        missing_features = [feature for feature in feature_names if feature not in input_data.columns]

        if missing_features:
            st.error(f"Missing required columns: {', '.join(missing_features)}")
        else:
            predictions = predict(
                model,
                scaler,
                feature_names,
                input_data,
                metadata["best_model_type"],
            )
            st.dataframe(predictions, use_container_width=True)
            st.download_button(
                "Download Predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

with tab_manual:
    st.write("Enter one transaction's values. The Kaggle dataset uses PCA features V1 to V28.")
    values = {}

    cols = st.columns(3)
    for index, feature in enumerate(feature_names):
        default_value = 0.0
        if feature == "Amount":
            default_value = 100.0
        values[feature] = cols[index % 3].number_input(feature, value=default_value, format="%.6f")

    if st.button("Predict Transaction"):
        manual_df = pd.DataFrame([values])
        prediction = predict(model, scaler, feature_names, manual_df, metadata["best_model_type"])
        probability = prediction.loc[0, "Fraud Probability"]
        label = prediction.loc[0, "Prediction"]

        if label == "Fraud":
            st.error(f"Prediction: Fraud transaction. Probability: {probability:.4f}")
        else:
            st.success(f"Prediction: Genuine transaction. Probability: {probability:.4f}")
>>>>>>> 34f56550a3901728f7f071a3a4ac63ab83d8a66e
