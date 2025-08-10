# steps/preprocess.py
import pandas as pd
import argparse
from pathlib import Path
import uuid
import time
import numpy as np

def preprocess(raw_input_dir: str, clean_output_dir: str):
    raw_input_path = Path(raw_input_dir)
    clean_output_path = Path(clean_output_dir)
    clean_output_path.mkdir(parents=True, exist_ok=True)

    cleaned_paths = []

    for file in raw_input_path.glob("*.csv"):
        df = pd.read_csv(file, index_col=False)
        df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        df['event_time'] = pd.to_datetime('now')  # current timestamp

        string_cols = df.select_dtypes(include='object').columns
        df[string_cols] = df[string_cols].fillna('unknown')

        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            if col in ["age", "income", "spending_score"]:
                df[col] = df[col].apply(lambda x: np.nan if x < 0 or pd.isna(x) else x)
                valid_mean = df[col].mean()
                df[col] = df[col].fillna(valid_mean)
            else:
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
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    preprocess(args.input_dir, args.output_dir)