import os
import pandas as pd
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)  # This will be /opt/ml/processing/output
    target_col = "recommended_card"

    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.csv"):
        df = pd.read_csv(file)

        print(f"Processing file: {file.name}")
        print("Separating features and labels...")

        # --- Light preprocessing ---
        # 1. Strip column names (avoid accidental spaces)
        df.columns = df.columns.str.strip()

        # 2. Coerce numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # invalid to NaN
        
        # Convert credit_score to float
        df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
        df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())

        # 3. Optional: Fill NaNs for numeric and categorical
        df[numeric_cols] = df[numeric_cols].fillna(0)
        categorical_cols = df.select_dtypes(include=["object"]).columns
        df[categorical_cols] = df[categorical_cols].fillna("unknown")

        # Separate features and labels
        X = df.drop(columns=[target_col])
        y = df[[target_col]]  # keep as DataFrame

        # Save in Parquet format under output directory
        X.to_parquet(output_path / f"[{file.stem}]_features.parquet", index=False)
        y.to_parquet(output_path / f"[{file.stem}]_labels.parquet", index=False)

        print(f"✅ Saved: {file.stem}_features.parquet and {file.stem}_labels.parquet")
