"""Utility file containing a notebook generator for the capstone notebook.

Run:
    python notebook_content.py
"""

import json
import textwrap
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


cells = [
    markdown(
        """
        # Credit Card Fraud Detection Using ML and DL

        This notebook builds machine learning and deep learning models to detect fraudulent credit card transactions.
        """
    ),
    markdown("## 1. Importing Libraries"),
    code(
        """
        import warnings
        warnings.filterwarnings("ignore")

        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from imblearn.over_sampling import SMOTE
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
        """
    ),
    markdown("## 2. Setting Project Paths"),
    code(
        """
        PROJECT_DIR = Path.cwd()
        DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
        MODEL_DIR = PROJECT_DIR / "models"
        RESULT_DIR = PROJECT_DIR / "results"

        MODEL_DIR.mkdir(exist_ok=True)
        RESULT_DIR.mkdir(exist_ok=True)

        RANDOM_STATE = 42
        """
    ),
    markdown("## 3. Preparing Dataset"),
    code(
        """
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}. Download the Kaggle credit card fraud CSV "
                "and save it as data/creditcard.csv"
            )

        df = pd.read_csv(DATA_PATH)
        df.head()
        """
    ),
    markdown("## 4. Dataset Overview"),
    code(
        """
        print("Dataset shape:", df.shape)
        print("\\nColumn names:")
        print(df.columns.tolist())
        print("\\nMissing values:")
        print(df.isnull().sum())
        print("\\nClass distribution:")
        print(df["Class"].value_counts())
        """
    ),
    markdown("## 5. Visualizing Class Imbalance"),
    code(
        """
        plt.figure(figsize=(5, 4))
        sns.countplot(data=df, x="Class")
        plt.title("Class Distribution: Genuine vs Fraud")
        plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
        plt.ylabel("Count")
        plt.show()
        """
    ),
    markdown("## 6. Feature and Target Selection"),
    code(
        """
        X = df.drop("Class", axis=1)
        y = df["Class"]

        feature_names = X.columns.tolist()
        print("Total features:", len(feature_names))
        print("Target column: Class")
        """
    ),
    markdown("## 7. Train Test Split"),
    code(
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print("Training data:", X_train.shape)
        print("Testing data:", X_test.shape)
        """
    ),
    markdown("## 8. Feature Scaling"),
    code(
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Scaling completed.")
        """
    ),
    markdown("## 9. Handling Imbalanced Data With SMOTE"),
    code(
        """
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        print("Before SMOTE:")
        print(y_train.value_counts())
        print("\\nAfter SMOTE:")
        print(pd.Series(y_train_balanced).value_counts())
        """
    ),
    markdown("## 10. Defining Evaluation Functions"),
    code(
        """
        def calculate_metrics(model_name, model_type, y_true, y_pred, y_probability):
            return {
                "model_name": model_name,
                "model_type": model_type,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_probability),
                "pr_auc": average_precision_score(y_true, y_probability),
            }


        def plot_confusion_matrix(model_name, y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"{model_name} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
        """
    ),
    markdown("## 11. Training Machine Learning Models"),
    code(
        """
        ml_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        }

        trained_ml_models = {}
        all_metrics = []

        for model_name, model in ml_models.items():
            print(f"Training {model_name}...")
            model.fit(X_train_balanced, y_train_balanced)

            y_pred = model.predict(X_test_scaled)
            y_probability = model.predict_proba(X_test_scaled)[:, 1]

            trained_ml_models[model_name] = model
            all_metrics.append(calculate_metrics(model_name, "machine_learning", y_test, y_pred, y_probability))

            print(classification_report(y_test, y_pred, zero_division=0))
            plot_confusion_matrix(model_name, y_test, y_pred)
        """
    ),
    markdown("## 12. Building Deep Learning Model"),
    code(
        """
        tf.random.set_seed(RANDOM_STATE)

        dl_model = Sequential([
            Input(shape=(X_train_balanced.shape[1],)),
            Dense(64, activation="relu"),
            Dropout(0.30),
            Dense(32, activation="relu"),
            Dropout(0.20),
            Dense(1, activation="sigmoid"),
        ])

        dl_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        dl_model.summary()
        """
    ),
    markdown("## 13. Training Deep Learning Model"),
    code(
        """
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        history = dl_model.fit(
            X_train_balanced,
            y_train_balanced,
            validation_split=0.2,
            epochs=30,
            batch_size=256,
            callbacks=[early_stopping],
            verbose=1,
        )
        """
    ),
    markdown("## 14. Evaluating Deep Learning Model"),
    code(
        """
        dl_probability = dl_model.predict(X_test_scaled).ravel()
        dl_pred = (dl_probability >= 0.5).astype(int)

        print(classification_report(y_test, dl_pred, zero_division=0))
        plot_confusion_matrix("Deep Learning", y_test, dl_pred)

        all_metrics.append(calculate_metrics("Deep Learning", "deep_learning", y_test, dl_pred, dl_probability))
        """
    ),
    markdown("## 15. Comparing All Models"),
    code(
        """
        comparison_df = pd.DataFrame(all_metrics).sort_values(by="f1_score", ascending=False)
        comparison_df
        """
    ),
    markdown("## 16. Saving Results"),
    code(
        """
        comparison_path = RESULT_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Model comparison saved to {comparison_path}")
        """
    ),
    markdown("## 17. Saving Best Model"),
    code(
        """
        best_row = comparison_df.iloc[0]
        best_model_name = best_row["model_name"]
        best_model_type = best_row["model_type"]

        print("Best model:", best_model_name)
        print("Best model type:", best_model_type)

        if best_model_type == "deep_learning":
            dl_model.save(MODEL_DIR / "best_deep_learning_model.keras")
        else:
            best_model = trained_ml_models[best_model_name]
            joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

        metadata = {
            "best_model_name": best_model_name,
            "best_model_type": best_model_type,
            "feature_names": feature_names,
            "scaler": scaler,
            "metrics": comparison_df.to_dict(orient="records"),
        }

        joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")
        print("Best model artifacts saved in the models folder.")
        """
    ),
    markdown("## 18. Testing Saved Model Artifacts"),
    code(
        """
        saved_metadata = joblib.load(MODEL_DIR / "model_metadata.joblib")
        sample_transaction = X_test.iloc[[0]]
        sample_scaled = saved_metadata["scaler"].transform(sample_transaction[saved_metadata["feature_names"]])

        if saved_metadata["best_model_type"] == "deep_learning":
            loaded_model = tf.keras.models.load_model(MODEL_DIR / "best_deep_learning_model.keras")
            fraud_probability = loaded_model.predict(sample_scaled).ravel()[0]
        else:
            loaded_model = joblib.load(MODEL_DIR / "best_model.joblib")
            fraud_probability = loaded_model.predict_proba(sample_scaled)[:, 1][0]

        prediction = "Fraud" if fraud_probability >= 0.5 else "Genuine"
        print("Fraud probability:", fraud_probability)
        print("Prediction:", prediction)
        """
    ),
    markdown(
        """
        ## 19. Streamlit Deployment

        After running this notebook successfully, start the app with:

        ```bash
        streamlit run app.py
        ```
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("Credit_Card_Fraud_Detection.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print("Created Credit_Card_Fraud_Detection.ipynb")
