from sklearn.metrics import classification_report
import joblib
from data_preprocessing import preprocess_data

model = joblib.load("models/churn_model.pkl")

_, X_test, _, y_test = preprocess_data("data/churn.csv")

preds = model.predict(X_test)

print(classification_report(y_test, preds))
