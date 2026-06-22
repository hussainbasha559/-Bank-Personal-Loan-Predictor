import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model    = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

class CustomerData(BaseModel):
    Age: int
    Experience: int
    Income: int
    Family: int
    Education: int
    Mortgage: int
    CD_Account: int

@app.post("/predict")
def predict(data: CustomerData):
    input_data = pd.DataFrame([{
        "Age":        data.Age,
        "Experience": data.Experience,
        "Income":     data.Income,
        "Family":     data.Family,
        "Education":  data.Education,
        "Mortgage":   data.Mortgage,
        "CD Account": data.CD_Account
    }])

    input_data  = input_data[features]
    scaled      = scaler.transform(input_data)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "prediction": int(prediction),
        "result":     "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved",
        "confidence": f"{probability*100:.1f}%"
    }

@app.get("/")
def home():
    return {"message": "Bank Loan API is running!"}