import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import argparse

def evaluate_model(input_dir, output_dir, artifacts_dir, label_col="recommended_card"):
    input_path = Path(input_dir)
    artifacts_path = Path(artifacts_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load artifacts
    model = joblib.load(artifacts_path / "model.joblib")
    preprocessor = joblib.load(artifacts_path / "preprocessor.joblib")
    label_encoder = joblib.load(artifacts_path / "label_encoder.joblib")

    # Load features & labels from parquet
    feature_files = sorted(list(input_path.glob("*_features.parquet")))
    label_files   = sorted(list(input_path.glob("*_labels.parquet")))

    dfs = []
    for feat_file, label_file in zip(feature_files, label_files):
        X_part = pd.read_parquet(feat_file)
        y_part = pd.read_parquet(label_file)
        dfs.append((X_part, y_part))

    X = pd.concat([x for x, _ in dfs], ignore_index=True)
    y = pd.concat([y for _, y in dfs], ignore_index=True)

    # Ensure we only use the correct label column
    y = y[label_col]

    # Transform labels to match training encoding
    y_encoded = label_encoder.transform(y)

    # Predict
    X_transformed = preprocessor.transform(X)
    y_pred = model.predict(X_transformed)

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_encoded, y_pred), 4),
        "precision": round(precision_score(y_encoded, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_encoded, y_pred, average="weighted"), 4),
        "f1_score": round(f1_score(y_encoded, y_pred, average="weighted"), 4)
    }

    # Save metrics
    pd.DataFrame([metrics]).to_json(output_path / "results.json", orient="records", indent=4)

    print(f"\n📊 Evaluation Results for: {label_col}")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--artifacts-dir", type=str, default="/opt/ml/processing/artifacts")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    evaluate_model(args.input_dir, args.output_dir, args.artifacts_dir)