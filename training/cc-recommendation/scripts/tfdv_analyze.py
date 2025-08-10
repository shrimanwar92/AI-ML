# tfdv_analyze.py
import pandas as pd
import tensorflow_data_validation as tfdv
import argparse
import os
from pathlib import Path
import glob

def combine_csv(all_files):
    list_of_dfs = []
    
    for filename in all_files:
        single_df = pd.read_csv(filename)
        list_of_dfs.append(single_df)

    return pd.concat(list_of_dfs, ignore_index=True)

def run_tfdv(input_dir, output_dir, create_schema, schema_dir):
    input_data_path = Path(input_dir)
    output_path = Path(output_dir)
    
    csvs = glob.glob(f"{input_data_path}/*.csv")

    # If no baseline schema, generate it
    if create_schema == "true":
        stats_path = output_path / "baseline_stats.txt"
        schema_path = output_path / "schema.pbtxt"
        combine_df = combine_csv(csvs)
        stats = tfdv.generate_statistics_from_dataframe(combine_df)
        tfdv.write_stats_text(stats, stats_path)
        schema = tfdv.infer_schema(stats)
        tfdv.write_schema_text(schema, schema_path)
    else:
        schema_path = Path(schema_dir)
        # Load existing schema and compare
        for csv in csvs:
            file_path = Path(csv)
            anomalies_path = output_path / f"[{file_path.name}]_anomalies.pbtxt"
            schema = tfdv.load_schema_text(f"{schema_path}/schema.pbtxt")
            df = pd.read_csv(csv)
            stats = tfdv.generate_statistics_from_dataframe(df)
            anomalies = tfdv.validate_statistics(stats, schema)
            tfdv.write_anomalies_text(anomalies, anomalies_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--create-schema", type=str)
    parser.add_argument("--schema-dir", type=str)

    args = parser.parse_args()

    run_tfdv(args.input_dir, args.output_dir, args.create_schema, args.schema_dir)