# 💧 Water Potability Classification | Machine Learning

**Author:** Soumya Shah | [GitHub](https://github.com/SoumyaShahh)

> A multi-model machine learning classification project predicting **whether water is safe to drink** based on 9 chemical features — trained on 3,276 water samples with comprehensive hyperparameter tuning, class imbalance handling, and comparative model evaluation.

---

## 📌 Project Overview

Access to safe drinking water is a critical public health challenge. This project builds a binary classification system to predict water potability based on chemical composition — enabling faster, data-driven water quality assessment without requiring full lab analysis for every sample.

**Business Problem:**
Traditional water safety testing is time-consuming and expensive. A machine learning model trained on chemical characteristics can provide a rapid first-pass safety classification — flagging unsafe water for further testing and prioritizing resources where they matter most.

---

## 🔢 Dataset Overview

| Metric | Value |
|---|---|
| Total Samples | 3,276 |
| Features | 9 chemical properties |
| Target | Binary (Potable = 1, Not Potable = 0) |
| Potable Samples | 1,278 (39%) |
| Non-Potable Samples | 1,998 (61%) |
| Class Imbalance | Yes — handled with SMOTE & RandomOverSampler |

**Chemical Features:**
| Feature | Description |
|---|---|
| pH | Acidity/alkalinity level (WHO safe range: 6.5–8.5) |
| Hardness | Calcium and magnesium concentration |
| Solids | Total dissolved solids (TDS) |
| Chloramines | Chlorine + ammonia disinfectant levels |
| Sulfate | Dissolved sulfate concentration |
| Conductivity | Ionic concentration indicator |
| Organic Carbon | Total organic carbon from natural sources |
| Trihalomethanes | Byproducts of chlorine disinfection |
| Turbidity | Clarity of water |

---

## ⚙️ Data Preprocessing Pipeline

```
Raw Data (3,276 samples)
    │
    ▼
Missing Value Handling
    │  • pH: 491 nulls → median imputation
    │  • Sulfate: 781 nulls → median imputation
    │  • Trihalomethanes: 162 nulls → median imputation
    ▼
Exploratory Data Analysis
    │  • Correlation heatmap
    │  • Distribution plots (histograms + boxplots)
    │  • Class imbalance analysis
    │  • Skewness detection (Solids, Organic Carbon)
    ▼
Feature Scaling
    │  • StandardScaler applied to all 9 features
    ▼
Class Imbalance Handling
    │  • RandomOverSampler → 1,998 : 1,998 balanced
    │  • SMOTE → 1,998 : 1,998 balanced
    ▼
Train/Test Split (70/30)
    └── Training: 2,293 samples
    └── Testing:    983 samples
```

---

## 🤖 Model Comparison

Five classifiers trained and evaluated — all with hyperparameter tuning:

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **Random Forest (GridSearchCV)** | **68.6%** | **65.3%** | **68.0%** | **68.6%** |
| Random Forest (Default) | 67.8% | 65.5% | 66.6% | 67.8% |
| K-Nearest Neighbors (k=20) | 66.5% | 61.0% | 66.3% | 66.5% |
| Decision Tree (GINI) | 66.9% | 61.3% | 67.3% | 66.9% |
| Decision Tree (Entropy) | 65.1% | 59.0% | 63.8% | 65.1% |
| Gaussian Naive Bayes | 62.8% | 61.3% | 67.3% | 66.9% |
| Bernoulli Naive Bayes | 62.6% | 57.9% | 59.9% | 63.0% |
| Logistic Regression | 62.8% | — | — | — |

**Winner: Random Forest with GridSearchCV** — 68.6% accuracy after tuning across 162 parameter combinations (486 total fits with 3-fold CV).

---

## 🔧 Hyperparameter Tuning

### Random Forest — GridSearchCV
```python
param_grid = {
    'max_depth': [10, 20, 30],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 200, 300]
}
# 162 combinations × 3-fold CV = 486 total fits
```

### KNN — GridSearchCV
```python
# Tested k = 1 to 49
# Optimal k = 20 (accuracy peaks at 0.65)
```

### Gaussian Naive Bayes — RandomizedSearchCV
```python
# var_smoothing tuned across 100 log-spaced values
# Best: var_smoothing = 0.081
```

---

## 📊 Key Findings

| Finding | Detail |
|---|---|
| **Best Model** | Random Forest — 68.6% accuracy after GridSearchCV |
| **Top Predictive Features** | pH, Sulfate, Chloramines |
| **Class Imbalance** | 61% non-potable vs 39% potable — handled with SMOTE & RandomOverSampler |
| **Skewness Detected** | Solids and Organic Carbon show right-skewed distributions |
| **Logistic Regression Limitation** | Near-zero recall for potable class — linear boundary insufficient |
| **SMOTE Impact** | Balanced classes improved minority class recall significantly |
| **Decision Tree Criterion** | GINI (66.9%) outperformed Entropy (65.1%) |

---

## 💡 Why ~68% Accuracy?

Water potability is notoriously difficult to classify from chemical features alone because:
- Chemical ranges for safe/unsafe water **overlap significantly**
- The dataset has **inherent noise** — water safety depends on combined interactions, not individual thresholds
- No single chemical feature strongly separates the two classes (low correlation with target)

This is a known challenge in water quality ML research — 68% represents a **strong baseline** given the feature complexity.

---

## 🚀 Future Enhancements

- **Gradient Boosting / XGBoost** — likely to push accuracy to 72–75%
- **Flask/Streamlit deployment** — real-time water safety prediction API
- **Feature engineering** — interaction terms between pH × Chloramines, Sulfate × Hardness
- **IoT integration** — real-time sensor data ingestion for continuous monitoring
- **Expanded dataset** — geo-tagged samples for regional water safety mapping

---

## 🛠️ Tech Stack

| Tool | Usage |
|---|---|
| Python | End-to-end ML pipeline |
| Pandas, NumPy | Data loading, cleaning, imputation |
| Matplotlib, Seaborn | EDA visualizations, correlation heatmap |
| scikit-learn | 5 ML models, StandardScaler, GridSearchCV, RandomizedSearchCV |
| imbalanced-learn | SMOTE, RandomOverSampler for class balancing |
| Graphviz, pydotplus | Decision tree visualization |
| Jupyter Notebook | Interactive development environment |

---

## 📁 Repository Structure

```
Water-Potability/
├── Water Potability Classification.ipynb   # Full ML pipeline with outputs
└── README.md
```

---

*Built by Soumya Shah | [GitHub](https://github.com/SoumyaShahh)*
