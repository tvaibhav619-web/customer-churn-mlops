import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from data_preprocessing import preprocess_data

# ---------------- Paths ----------------
DATA_PATH = "data/churn.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_pipeline.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- MLflow Config ----------------
mlflow.set_experiment("Customer Churn Prediction")

# ---------------- Load Data ----------------
preprocessor, X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)

# ---------------- Model ----------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ---------------- Training + Tracking ----------------
with mlflow.start_run():
    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # -------- Log params --------
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # -------- Log metrics --------
    mlflow.log_metric("accuracy", accuracy)

    # -------- Log model --------
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model"
    )

    # -------- Save locally --------
    joblib.dump(pipeline, MODEL_PATH)

    print(f"✅ Model trained & saved at {MODEL_PATH}")
    print(f"📊 Accuracy: {accuracy:.4f}")
