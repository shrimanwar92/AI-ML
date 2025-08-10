# evaluate_batch.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(X_path, y_path, label=""):
    model = joblib.load("artifacts/model.joblib")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)["income"]
    X_transformed = preprocessor.transform(X)
    y_pred = model.predict(X_transformed)

    print(f"\n📊 Evaluation on: {label}")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall:    {recall_score(y, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred):.4f}")

if __name__ == "__main__":
    evaluate_model("data/test_clean.csv", "data/test_clean_labels.csv", "Clean Data")
    evaluate_model("data/test_degraded.csv", "data/test_degraded_labels.csv", "Degraded Data")
