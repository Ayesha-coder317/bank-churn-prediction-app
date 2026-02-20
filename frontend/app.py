import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦")
st.title("🏦 Bank Churn Prediction")
st.write("Fill the customer details and click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    CreditScore = st.number_input("CreditScore", min_value=300, max_value=900, value=650)
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=40)
    Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)

with col2:
    Balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
    Num_Of_Products = st.number_input("Num Of Products", min_value=1, max_value=4, value=2)
    Has_Credit_Card = st.selectbox("Has Credit Card", [0, 1], index=1)
    Is_Active_Member = st.selectbox("Is Active Member", [0, 1], index=1)
    Estimated_Salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

payload = {
    "CreditScore": int(CreditScore),
    "Geography": Geography,
    "Gender": Gender,
    "Age": int(Age),
    "Tenure": int(Tenure),
    "Balance": float(Balance),
    "Num_Of_Products": int(Num_Of_Products),
    "Has_Credit_Card": int(Has_Credit_Card),
    "Is_Active_Member": int(Is_Active_Member),
    "Estimated_Salary": float(Estimated_Salary),
}

if st.button("Predict"):
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        res.raise_for_status()
        pred = res.json().get("churn_prediction")

        if pred == 1:
            st.error("Prediction: Churn = YES (Customer likely to leave)")
        else:
            st.success("Prediction: Churn = NO (Customer likely to stay)")

        st.caption(f"API response: {res.json()}")

    except Exception as e:
        st.error(f"Error calling API: {e}")
        st.write("Make sure FastAPI is running at http://127.0.0.1:8000")
