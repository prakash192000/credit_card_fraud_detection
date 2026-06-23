# Credit Card Fraud Detection

This capstone project builds both machine learning and deep learning models to detect fraudulent credit card transactions. The main project work is in a Jupyter notebook, and the trained best model is deployed through a Streamlit UI.

## Dataset

Download the Kaggle dataset and place the CSV here:

```text
data/creditcard.csv
```

The expected target column is:

```text
Class
```

where `0` means normal transaction and `1` means fraud transaction.

## Setup

```bash
pip install -r requirements.txt
```

## Step 1: Run Notebook

Open and run:

```text
Credit_Card_Fraud_Detection.ipynb
```

The notebook trains ML and DL models, compares the metrics, and saves the best model in `models/`.

## Step 2: Run Streamlit UI

```bash
streamlit run app.py
```

## Models Included

- Logistic Regression
- Random Forest
- Gradient Boosting
- Deep Learning Neural Network

## Outputs

The project saves:

- ML model files in `models/`
- DL model file in `models/`
- Metrics comparison in `results/model_comparison.csv`
- Best model metadata in `models/model_metadata.joblib`

## Main Evaluation Metrics

Because fraud detection is an imbalanced classification problem, the most important metrics are:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Accuracy is included but should not be the main deciding metric.
