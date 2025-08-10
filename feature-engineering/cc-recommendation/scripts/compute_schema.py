import tensorflow_data_validation as tfdv
from pathlib import Path
import glob
import pandas as pd
import argparse

def combine_csv(all_files):
    list_of_dfs = []
    
    for filename in all_files:
        single_df = pd.read_csv(filename)
        list_of_dfs.append(single_df)

    return pd.concat(list_of_dfs, ignore_index=True)

def compute_schema(baseline_input_dir, output_dir):
    baseline_data_path = Path(baseline_input_dir)
    output_path = Path(output_dir)
    stats_path = output_path / "baseline_stats.txt"
    schema_path = output_path / "schema.pbtxt"

    csvs = glob.glob(f"{baseline_data_path}/*.csv")
    combine_df = combine_csv(csvs)
    
    # Load data
    stats = tfdv.generate_statistics_from_dataframe(combine_df)

    # Infer schema
    schema = tfdv.infer_schema(statistics=stats)
    tfdv.display_schema(schema)

    # Save schema
    tfdv.write_stats_text(stats, stats_path)
    tfdv.write_schema_text(schema, schema_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    compute_schema(args.baseline_input_dir, args.output_dir)
