# evaluate_realtime.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report

model = joblib.load("artifacts/model.joblib")
preprocessor = joblib.load("artifacts/preprocessor.joblib")

def predict_single(row):
    df = pd.DataFrame([row])
    X_transformed = preprocessor.transform(df)
    return model.predict(X_transformed)[0]

# Load degraded data
df = pd.read_csv("data/test_degraded.csv")
y_true = pd.read_csv("data/test_degraded_labels.csv")["income"]
y_pred = []

for _, row in df.iterrows():
    pred = predict_single(row)
    y_pred.append(pred)

print("\n⚡ Real-Time (Simulated) on Degraded Data:")
print(classification_report(y_true, y_pred))
