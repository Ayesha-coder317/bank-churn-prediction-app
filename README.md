# 🏦 Bank Churn Prediction App  
FastAPI + Streamlit + scikit-learn

## 📌 Project Overview
This project demonstrates an end-to-end Machine Learning deployment pipeline for predicting bank customer churn.

The system includes:
- Model training using scikit-learn
- FastAPI backend to serve predictions
- Streamlit frontend for user interaction

The application allows users to input customer details and receive a churn prediction (0 = Stay, 1 = Leave).

---

## 📊 Dataset
Bank Churn Modelling Dataset

**Target Variable:**  
`Churn`  
- 0 → Customer stays  
- 1 → Customer churns  

**Features Used:**
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- Num Of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

---

## 🧠 Model Training
- Algorithm: Logistic Regression
- Preprocessing:
  - OneHotEncoding for categorical features (Geography, Gender)
  - Numeric features passed directly
- Train/Test Split: 80/20 (Stratified)
- Model saved as: `model.pkl`

---

## 🚀 Backend (FastAPI)

### Start Backend:
```bash
cd backend
source ../training/.venv/bin/activate
uvicorn app.main:app --reload
