# steps/preprocess.py
import os
import pandas as pd
import argparse
from pathlib import Path

def preprocess(raw_input_dir: str, clean_output_dir: str):
    raw_input_path = Path(raw_input_dir)
    clean_output_path = Path(clean_output_dir)
    clean_output_path.mkdir(parents=True, exist_ok=True)

    cleaned_paths = []

    for file in raw_input_path.glob("*.csv"):
        df = pd.read_csv(file, index_col=False)
        df.rename(columns={
            'education.num': 'education_num',
            'marital.status': 'marital_status',
            'capital.gain': 'capital_gain',
            'capital.loss': 'capital_loss',
            'hours.per.week': 'hours_per_week',
            'native.country': 'native_country'
        }, inplace=True)

        string_cols = df.select_dtypes(include='object').columns
        df[string_cols] = df[string_cols].fillna('unknown')

        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

        cleaned_file = clean_output_path / file.name
        df.to_csv(cleaned_file, index=False)
        cleaned_paths.append(str(cleaned_file))
        print(len(df))
        print(df.head())
        print(f"Preprocessing complete for: {file.name}")

    return cleaned_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()
    print(args)

    preprocess(args.input_dir, args.output_dir)