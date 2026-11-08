# Machine Learning Classification & Evaluation Dashboard

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare five distinct Machine Learning classification algorithms on a multi-feature dataset. The trained models are deployed as an interactive Streamlit application on Streamlit Community Cloud to allow dynamic inference and evaluation on unseen test datasets.

## b. Dataset Description
* **Features**: 14 continuous and categorical features representing domain indicators.
* **Instance Count**: 1,000 samples (800 Training / 200 Test split).
* **Target Variable**: Binary target label (`0` or `1`).

## c. GitHub Repository Link
- Repository URL: `https://github.com/kvnsharshavardhan/ml-classification-streamlit-app`
- Live Streamlit App URL: `https://ml-classification-app-app-rp4q3e7e8auvqe4mbyyhgn.streamlit.app/`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8650 | 0.9320 | 0.8540 | 0.8750 | 0.8643 | 0.7302 |
| **Decision Tree** | 0.8100 | 0.8550 | 0.8000 | 0.8200 | 0.8098 | 0.6203 |
| **KNN** | 0.8450 | 0.9100 | 0.8350 | 0.8500 | 0.8424 | 0.6901 |
| **Naive Bayes** | 0.8300 | 0.9020 | 0.8200 | 0.8400 | 0.8299 | 0.6601 |
| **Support Vector Machine** | 0.8800 | 0.9450 | 0.8700 | 0.8900 | 0.8799 | 0.7602 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly due to linear feature separation after scaling; provided stable probability estimates with minimal overfitting. |
| **Decision Tree** | Fast computation but slightly underperformed compared to ensemble/margin-based classifiers due to axis-aligned decision boundaries. |
| **KNN** | Showed high precision when features were normalized; performance is sensitive to distance metrics and outlier noise. |
| **Naive Bayes** | Efficient baseline model; slightly lower metrics due to the strong independence assumption among correlated features. |
| **Support Vector Machine** | Delivered the highest overall accuracy and MCC score using an RBF kernel, effectively handling complex feature interactions. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the highest performance across all evaluation metrics (Accuracy: 0.8800, AUC: 0.9450, MCC: 0.7602). |
