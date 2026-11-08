import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation Dashboard", layout="wide")

st.title("📊 Multi-Model Classification Evaluation Dashboard")
st.markdown("BITS Pilani WILP - Machine Learning Assignment 2")

st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_dict = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_dict.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("### 📋 Uploaded Test Dataset Preview")
    st.dataframe(test_df.head())

    if "target" not in test_df.columns:
        st.error("Error: Uploaded CSV must contain the target column labeled 'target'.")
    else:
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        try:
            scaler = joblib.load("model/scaler.pkl")
            model = joblib.load(model_dict[selected_model_name])
            
            if selected_model_name == "Decision Tree":
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            else:
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

            st.write(f"## 📈 Performance Metrics: {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")
            
            col_cm, col_report = st.columns(2)

            with col_cm:
                st.write("### 🔲 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                            xticklabels=["Class 0", "Class 1"], 
                            yticklabels=["Class 0", "Class 1"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col_report:
                st.write("### 📄 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0))

        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")
else:
    st.info("👈 Please upload `test_data.csv` from the sidebar to evaluate the models.")
