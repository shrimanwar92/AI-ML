# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load UCI Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

# Drop missing
df.dropna(inplace=True)

# Binary classification label
df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

# Feature/label split
X = df.drop("income", axis=1)
y = df["income"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
num_features = X.select_dtypes(include="number").columns.tolist()
cat_features = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
])

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save model and preprocessor
joblib.dump(pipeline.named_steps["clf"], "artifacts/model.joblib")
joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

# Save clean test set
X_test.to_csv("data/test_clean.csv", index=False)
y_test.to_frame().to_csv("data/test_clean_labels.csv", index=False)

# Simulate degraded version: change ages and income/education
X_test_degraded = X_test.copy()
X_test_degraded["age"] = X_test_degraded["age"] + np.random.randint(10, 30, size=len(X_test_degraded))
X_test_degraded["education"] = "Preschool"
X_test_degraded["hours-per-week"] = 1

X_test_degraded.to_csv("data/test_degraded.csv", index=False)
y_test.to_frame().to_csv("data/test_degraded_labels.csv", index=False)

print("✅ Model trained and test datasets saved.")
