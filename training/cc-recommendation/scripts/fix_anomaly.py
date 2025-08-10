# steps/preprocess.py
import pandas as pd
import argparse
from pathlib import Path
import uuid
import time
import tensorflow_data_validation as tfdv
import json
import sys

def fix_anomalies(schema, input_dir: str, output_dir: str, anomalies_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    anomalies_path = Path(anomalies_dir)

    anomalies_path.mkdir(parents=True, exist_ok=True)  # ensure dir exists

    for file in input_path.glob("*.csv"):
        df = pd.read_csv(file, index_col=False)

        string_cols = df.select_dtypes(include='object').columns
        df[string_cols] = df[string_cols].fillna('unknown')

        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Convert credit_score to float
        df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
        df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())

        # Fix unexpected current_card values
        allowed_cards = ['CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'CC7', 'CC8', 'CC9', 'CC10']  # Example from schema
        df.loc[~df['current_card'].isin(allowed_cards), 'current_card'] = 'unknown'

        # Fix unexpected gender values
        allowed_genders = ['Male', 'Female', 'Other']
        df.loc[~df['gender'].isin(allowed_genders), 'gender'] = 'unknown'

        cleaned_file = output_path / file.name
        new_stats = tfdv.generate_statistics_from_dataframe(df)
        new_anomalies = tfdv.validate_statistics(new_stats, schema)
        anomaly_file_path = anomalies_path / f"[{file.name}]_anomalies.pbtxt"
        tfdv.write_anomalies_text(new_anomalies, str(anomaly_file_path))

        if not new_anomalies.anomaly_info:
            df.to_csv(cleaned_file, index=False)
            print(f"Clean csv saved in S3: {file.name}")
        else:
            print(f"Anomalies found. Please check {anomaly_file_path}")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--anomalies-dir", type=str)
    parser.add_argument("--schema-dir", type=str)
    args = parser.parse_args()

    schema = tfdv.load_schema_text(f"{args.schema_dir}/schema.pbtxt")

    fix_anomalies(schema, args.input_dir, args.output_dir, args.anomalies_dir)