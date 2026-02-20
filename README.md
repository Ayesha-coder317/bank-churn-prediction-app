# 🏦 Bank Churn Prediction App  
FastAPI + Streamlit + scikit-learn

---

## 📌 Project Overview
This project demonstrates an end-to-end Machine Learning deployment pipeline for predicting bank customer churn.

The system includes:
- Model training using scikit-learn
- FastAPI backend to serve predictions
- Streamlit frontend for user interaction

The application allows users to input customer details and receive a churn prediction:

- **0 → Customer likely to stay**
- **1 → Customer likely to churn**

This project follows a production-style ML architecture with clear separation of concerns between training, API serving, and frontend interaction.

---

## 📊 Dataset

**Dataset:** Bank Churn Modelling Dataset  

**Target Variable:**  
`Churn`
- 0 → Customer stays  
- 1 → Customer churns  

### Features Used:
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

The model was trained using:

- **Algorithm:** Logistic Regression
- **Pipeline:** scikit-learn `Pipeline`
- **Preprocessing:**
  - OneHotEncoding for categorical features (Geography, Gender)
  - Numeric features passed through unchanged
- **Train/Test Split:** 80/20 (Stratified)
- **Model Artifact:** Saved as `model.pkl`

The preprocessing and model are bundled together inside a single pipeline to ensure consistent inference during API calls.

---

## 🗂️ Project Structure

bank_churn_app/
├── backend/
│ ├── app/
│ │ ├── main.py
│ │ └── model.py
│ ├── model.pkl
│ └── requirements.txt
├── frontend/
│ ├── app.py
│ └── requirements.txt
├── training/
│ ├── train_model.py
│ └── bank_churn_modelling.csv
└── README.md


---

## 🚀 Backend (FastAPI)

The FastAPI backend loads `model.pkl` and exposes prediction endpoints.

### Start Backend

```bash
cd backend
source ../training/.venv/bin/activate
uvicorn app.main:app --reload
Backend runs at: http://127.0.0.1:8000

http://127.0.0.1:8000/docs - API documentation
example request:

{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.0,
  "Num_Of_Products": 1,
  "Has_Credit_Card": 1,
  "Is_Active_Member": 1,
  "Estimated_Salary": 101348.88
}
response : {
  "churn_prediction": 0
}
🖥️ Frontend (Streamlit)

The Streamlit app provides a user-friendly interface to interact with the API.

Start Frontend

Open a new terminal:
cd frontend
source ../training/.venv/bin/activate
streamlit run app.py
Frontend runs at: http://localhost:8501
Users can:

Enter customer details

Click Predict

Instantly see churn prediction results

🔄 End-to-End Workflow

User enters customer details in Streamlit.

Streamlit sends JSON request to FastAPI /predict.

FastAPI loads model.pkl.

Model processes input using trained pipeline.

Prediction returned to frontend.

UI displays result (Churn = Yes/No).

📦 Requirements

Backend dependencies:

fastapi

uvicorn

pandas

numpy

scikit-learn

Frontend dependencies:

streamlit

requests

