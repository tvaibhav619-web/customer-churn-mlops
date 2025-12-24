# Customer Churn Prediction – MLOps Project

## Overview
End-to-end MLOps project to predict customer churn with automated training,
experiment tracking, API deployment, and containerization.

## Tech Stack
- Python, Scikit-learn
- MLflow
- FastAPI
- Docker

## How to Run
1. Install dependencies
2. Train model: `python src/train.py`
3. Start API: `uvicorn api.app:app --reload`
4. Dockerize: `docker build -t churn-mlops .`
