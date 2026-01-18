import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_pred_gradient_boosting.pkl")
    gender_encoder = joblib.load("gender_label_encoder.pkl")
    geo_encoder = joblib.load("geography_oh.pkl")
    return model, gender_encoder, geo_encoder

model, gender_encoder, geo_encoder = load_artifacts()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #000000;
    color: #000000;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2,   {
    color: black;
}
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#3b82f6);
    color: white;
    border-radius: 12px;
    font-size: 18px;
    height: 3em;
    width: 100%;
}
.stMetric {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
}
.stSidebar {
    background-color: #1e293b;
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Project Overview")
st.sidebar.markdown("""
**Bank Customer Churn Prediction**

This project predicts whether a bank customer is likely to **leave the bank** (churn) based on customer information and account activity.

**Features Used:**
- Credit Score
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Ownership
- Active Membership
- Estimated Salary
- Geography

**Model:** Gradient Boosting Classifier

**Goal:** Help the bank **retain customers** by identifying high-risk accounts.

""")

# ---------------- HEADER ----------------
st.title("Bank Customer Churn Prediction")
st.markdown(
    "Use this app to predict whether a customer is **likely to churn** using a pre-trained Gradient Boosting model."
)
st.divider()

# ---------------- INPUTS ----------------
st.subheader("Enter Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)

with col2:
    balance = st.number_input("Account Balance", 0.0, 300000.0, 60000.0)
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    active_member = st.selectbox("Active Member", ["Yes", "No"])

with col3:
    salary = st.number_input("Estimated Salary", 0.0, 300000.0, 70000.0)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# ---------------- FEATURE PROCESSING ----------------
def prepare_input():
    gender_encoded = gender_encoder.transform([gender])[0]

    geo_encoded = geo_encoder.transform([[geography]])
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=geo_encoder.get_feature_names_out()
    )

    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [1 if has_card == "Yes" else 0],
        "IsActiveMember": [1 if active_member == "Yes" else 0],
        "EstimatedSalary": [salary]
    })

    final_df = pd.concat([input_df, geo_df], axis=1)
    return final_df

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):
    X = prepare_input()
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("Customer is Likely to Churn")
        else:
            st.success("Customer is Not Likely to Churn")


# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<center>Built with Streamlit • Gradient Boosting • ML • By Hafiz Rayyan Asif</center>",
    unsafe_allow_html=True
)
