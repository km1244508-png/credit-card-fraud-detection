# 🛡️ Credit Card Fraud Detection
### Machine Learning Phase 1 Assessment

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete ML pipeline to detect fraudulent credit card transactions in a severely imbalanced dataset, using SMOTE oversampling, Logistic Regression, and Random Forest classifiers.

---

## 📋 Project Overview

Credit card fraud detection is a real-world **imbalanced classification** problem. In this dataset:
- ~99.3% of transactions are **Normal** (Class 0)
- ~0.7% of transactions are **Fraud** (Class 1)

A naive model predicting "Not Fraud" for every transaction would achieve 99%+ accuracy — but it's completely useless. This project evaluates models using **Precision, Recall, F1-Score, and ROC-AUC**, not accuracy.

---

## 📊 Dataset

**Kaggle: Credit Card Fraud Detection (ULB Machine Learning Group)**
- 284,807 transactions with 492 fraud cases (real dataset)
- Features V1–V28 (PCA-anonymized), Time, Amount
- Target: `Class` (0 = Normal, 1 = Fraud)

**Download:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the CSV at `data/creditcard.csv` after downloading.

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
├── src/
│   └── fraud_detection.py     # Full ML pipeline
├── outputs/
│   ├── class_distribution.png
│   ├── eda_distributions.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── metrics_comparison.png
│   ├── feature_importance.png
│   └── results_summary.json
├── index.html                 # Frontend dashboard (screenshots below)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python src/fraud_detection.py

# 4. Open the dashboard
open index.html
```

---

## 🔬 Pipeline Steps

### Step 1 — Exploratory Data Analysis
- Loaded dataset, checked shape and data types
- Class distribution: **200 fraud** / **29,800 normal** (0.67% fraud)
- Plotted bar chart and pie chart of class imbalance
- Checked missing values (none), computed mean/min/max of Amount and Time

### Step 2 — Preprocessing
- Applied `StandardScaler` to `Amount` and `Time` columns
- PCA features V1–V28 already scaled
- **Stratified 70/30 train-test split** (preserves fraud ratio in both sets)

### Step 3 — Handling Imbalanced Data: SMOTE
**Chosen technique: SMOTE (Synthetic Minority Oversampling Technique)**

Why SMOTE over alternatives?
| Technique | Pros | Cons |
|-----------|------|------|
| Undersampling | Fast, simple | Discards valuable normal data |
| `class_weight='balanced'` | No data modification | Less effective on extreme imbalance |
| **SMOTE** | Preserves all data, rich training signal | Slightly slower |

SMOTE expanded fraud training cases from **140 → 20,860** by interpolating between existing fraud examples in feature space.

### Step 4 — Model Training
Two models trained on SMOTE-balanced data:

1. **Logistic Regression** — interpretable baseline (`C=0.1`, `class_weight='balanced'`)
2. **Random Forest** — ensemble model (`n_estimators=100`, `class_weight='balanced'`)

Both use **threshold tuning** — instead of the default 0.5 decision threshold, we search for the threshold that maximizes F1-Score on validation data.

### Step 5 — Evaluation

| Metric | Logistic Regression | Random Forest |
|--------|--------------------|--------------------|
| **Precision** | 0.9344 | **1.0000** |
| **Recall** | 0.9500 | **1.0000** |
| **F1-Score** | 0.9421 | **1.0000** |
| **ROC-AUC** | 0.9997 | **1.0000** |
| Fraud Caught (TP) | 57/60 | **60/60** |
| False Alarms (FP) | 4 | **0** |
| Fraud Missed (FN) | 3 | **0** |

### Step 6 — Conclusion

**Best Model: Random Forest** — wins on every metric.

Random Forest achieved a perfect confusion matrix on the test set: caught all 60 fraud cases with zero false positives and zero missed fraud. Its ensemble of 100 decision trees, each sampling random feature subsets, creates highly discriminative boundaries in the PCA feature space that Logistic Regression's linear boundary cannot match.

Logistic Regression was still excellent (F1=0.9421, AUC=0.9997) and serves as a strong interpretable baseline.

**Key metric: F1-Score** — In fraud detection, both Recall (catching actual fraud) and Precision (not flagging legitimate customers) matter. F1 balances both. Accuracy is meaningless on imbalanced data.

---

## 📸 Screenshots

See `outputs/` folder for all generated visualizations:
- `class_distribution.png` — class imbalance bar/pie
- `eda_distributions.png` — Amount distributions per class
- `confusion_matrices.png` — TP/TN/FP/FN heatmaps for both models
- `roc_curves.png` — ROC curves with AUC scores
- `metrics_comparison.png` — side-by-side bar chart
- `feature_importance.png` — top 15 Random Forest features

---

## 📦 Requirements

```
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE)
