from fastapi import FastAPI
from pydantic import BaseModel
from .model import predict_churn

app = FastAPI(title="Bank Churn Prediction API")


@app.get("/")
def home():
    return {"message": "Welcome to the Bank Churn Prediction API!"}


class ChurnRequest(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    Num_Of_Products: int
    Has_Credit_Card: int
    Is_Active_Member: int
    Estimated_Salary: float


@app.post("/predict")
def predict(req: ChurnRequest):
    features = req.model_dump()
    prediction = predict_churn(features)
    return {"churn_prediction": prediction}

