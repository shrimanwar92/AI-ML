import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def train(input_path: Path, output_path: Path):
    # ======================
    # 1. Load parquet files
    # ======================
    feature_files = sorted(list(input_path.glob("*_features.parquet")))
    label_files   = sorted(list(input_path.glob("*_labels.parquet")))

    if not feature_files:
        raise FileNotFoundError(f"No *_features.parquet files found in {input_path}")

    dfs = []
    for feat_file, label_file in zip(feature_files, label_files):
        X_part = pd.read_parquet(feat_file)
        y_part = pd.read_parquet(label_file)
        dfs.append((X_part, y_part))

    X = pd.concat([x for x, _ in dfs], ignore_index=True)
    y = pd.concat([y for _, y in dfs], ignore_index=True)

    # ======================
    # 2. Prepare labels
    # ======================
    label_col = "recommended_card"
    if label_col not in y.columns:
        raise ValueError(f"Label column '{label_col}' not found in labels file.")

    y = y[label_col]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save label encoder
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, output_path / "label_encoder.joblib")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ======================
    # 3. Preprocessing
    # ======================
    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(include="object").columns.tolist()

    # Remove accidental label col from features
    for col in [label_col]:
        if col in num_features:
            num_features.remove(col)
        if col in cat_features:
            cat_features.remove(col)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # Fit preprocessor separately (so we can save it)
    X_train_processed = preprocessor.fit_transform(X_train)

    # ======================
    # 4. Train model
    # ======================
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_processed, y_train)

    # ======================
    # 5. Save artifacts separately
    # ======================
    joblib.dump(preprocessor, output_path / "preprocessor.joblib")
    joblib.dump(clf, output_path / "model.joblib")

    # # Save clean test sets
    # X_test.to_csv(output_path / "test_clean.csv", index=False)
    # pd.DataFrame({"label": y_test}).to_csv(output_path / "test_clean_labels.csv", index=False)

    print(f"✅ Model and preprocessor saved separately to {output_path}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with *_features.parquet and *_labels.parquet files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save model and artifacts")
    args = parser.parse_args()

    train(input_path=Path(args.input_dir), output_path=Path(args.output_dir))
