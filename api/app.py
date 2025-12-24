from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# ---------------- App ----------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Pipeline-based Customer Churn Prediction",
    version="2.0"
)

# ---------------- Load Pipeline ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)

# ---------------- Input Schema ----------------
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    InternetService: str
    Contract: str
    MonthlyCharges: float
    TotalCharges: float

# ---------------- Routes ----------------
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API (Pipeline) is running 🚀",
        "docs": "Visit /docs for Swagger UI"
    }

@app.post("/predict")
def predict(data: ChurnInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict using pipeline
    prediction = pipeline.predict(input_df)

    return {
        "churn": int(prediction[0])
    }
