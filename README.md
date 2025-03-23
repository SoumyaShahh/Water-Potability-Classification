# Water Potability Classification using Machine Learning

This project focuses on predicting **whether water is safe to drink** based on chemical characteristics. Using a dataset with over 5K records, we applied five different machine learning models to classify water samples as **potable** or **non-potable**, using Python for analysis and model building.

---

## 📌 Project Overview

This project was developed as part of an academic course in data science. The goal was to evaluate **water quality** by predicting potability based on features such as:

- pH
- Hardness
- Sulfate
- Chloramines
- Conductivity

We explored and cleaned the data, performed feature engineering, trained multiple models, and compared their performance to determine the best-performing classifier.

---

## 🧰 Tools & Technologies

- Python (Jupyter Notebook)
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- GridSearchCV, RandomizedSearchCV

---

## 🤖 Models Used

1. **Logistic Regression** – Baseline linear classifier  
2. **K-Nearest Neighbors** – Distance-based algorithm  
3. **Decision Tree** – Simple rule-based classifier  
4. **Naive Bayes** – Probabilistic model using Gaussian/Bernoulli distributions  
5. **Random Forest** – Ensemble model with feature importance (**Best Accuracy: ~68%**)

---

## ⚙️ Data Processing & Evaluation

- Missing values handled via **median imputation**
- Feature scaling for distance-based models
- Binary encoding of target variable (`0 = Not Potable`, `1 = Potable`)
- Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- Hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`

---

## 📊 Key Insights

| Key Insight                            | Details                                                                 |
|----------------------------------------|-------------------------------------------------------------------------|
| Best Performing Model                  | **Random Forest** with ~68% accuracy                                    |
| Top Predictive Features                | **pH**, **Sulfate**, **Chloramines**                                    |
| Data Distribution                      | Dataset was **imbalanced**, requiring careful attention to **precision** and **recall** |
| Key EDA Observation                    | Skewness detected in **Solids** and **Organic Carbon**                  |

---

## 🧠 Python Skills Demonstrated

- Data wrangling and preprocessing using `pandas` and `numpy`
- Visual exploration using `matplotlib` and `seaborn`
- Model building and tuning using `scikit-learn`
- Custom function creation for evaluation and automation
- Feature importance analysis and comparative model performance plots

---

## 📁 Files Included

- `Group_10_MIS_545.ipynb` – Complete Jupyter Notebook with code, outputs, and charts  

---

## 🚀 Future Enhancements

- Deploy model using Flask/Streamlit
- Use SMOTE for class balancing
- Explore deep learning and boosting models for better accuracy
- Integrate real-world water quality data from IoT
