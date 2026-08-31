# Titanic Dataset: Multi-Model Survival Prediction

A machine learning project comparing multiple classification algorithms on the Titanic dataset, featuring data preprocessing, model evaluation, and an interactive Streamlit web application.

---

## Overview
This project evaluates multiple classification models using the Titanic dataset from Seaborn to predict passenger survival. The best-performing model (Support Vector Classifier) is exported as `model5.pkl` and deployed through a Streamlit user interface.

---

## Data Preprocessing & Features
* **Dropped Columns:** `deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`
* **Imputation:** Missing `age` values filled with the mean
* **Cleaning:** Dropped rows with missing `embarked` values
* **Encoding:** `LabelEncoder` applied to `sex`, `embarked`, and `alone`
* **Feature Scaling:** `StandardScaler` applied for distance- and margin-based models (KNN, Decision Tree, SVC)

### Feature Set (8 Features)
1. `pclass`
2. `sex`
3. `age`
4. `sibsp`
5. `parch`
6. `fare`
7. `embarked`
8. `alone`

---

## Model Benchmark & Results

| Model | Classifier | Accuracy | Notes |
| :--- | :--- | :---: | :--- |
| **Model 1** | Logistic Regression | 80.34% | Baseline classifier |
| **Model 2** | K-Nearest Neighbors (k=5) | 79.21% | Scaled features |
| **Model 3** | Gaussian Naive Bayes | 78.09% | Probabilistic model |
| **Model 4** | Decision Tree Classifier | 69.10% | Default tree depth |
| **Model 5** | **Support Vector Classifier (RBF Kernel)** | **82.58%** | **Best Performance (Saved as `model5.pkl`)** |

---

## Project Structure

```text
├── Multiple_Models.ipynb  # Jupyter notebook containing data pipeline and model training
├── model5.pkl             # Serialized best-performing SVC model
├── app.py                 # Interactive Streamlit application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
