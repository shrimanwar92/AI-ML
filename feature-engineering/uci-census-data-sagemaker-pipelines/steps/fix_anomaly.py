import sys
sys.path.append('/opt/ml/code')
from tfdv_validate import _save_anomalies_to_md, _save_anomalies_pbtxt
print("✅ Imported successfully")
import tensorflow_data_validation as tfdv
import pandas as pd
import contextlib
import os
import argparse
from pathlib import Path
import glob

# Function to get feature by name
def get_feature_by_name(schema, name):
    for feature in schema.feature:
        if feature.name == name:
            return feature
    return None

def clean_df_with_schema(data_path, file_name, schema, anomalies):
    print("Trying to fix anomalies...")
    df = pd.read_csv(data_path / file_name)

    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        #feature_schema = next((f for f in schema.feature if f.name == feature_name), None)
        feature_schema = get_feature_by_name(schema, feature_name)
        
        for reason in anomaly_info.reason:
            short_desc = reason.short_description.lower()
            description = reason.description.lower()
            print(f">>>>>>>>{feature_name}<<<<<<<<<<", short_desc)
            
            if "out-of-range" in short_desc and (feature_schema and feature_schema.int_domain):
                if feature_schema.int_domain.min:
                    min_expected = feature_schema.int_domain.min 
                    print(f"🧹 Clipping [{feature_name}] to min [{min_expected}]")
                    df.loc[df[feature_name] < min_expected, feature_name] = min_expected
                
                if feature_schema.int_domain.max:
                    max_expected = feature_schema.int_domain.max
                    print(f"🧹 Clipping [{feature_name}] to max [{max_expected}]")
                    df.loc[df[feature_name] > max_expected, feature_name] = max_expected

            if "int but got float" in description and (feature_schema and feature_schema.int_domain):
                print(f"🧹 Converting [{feature_name}] float to int")
                df[feature_name] = df[feature_name].astype(int)

    return df

def is_anomaly_present(anomalies):
    if len(anomalies.anomaly_info) > 0:
        return True
    return False


def fix_anomalies(schema_dir, data_dir, anomalies_dir, output_data, output_anomalies) -> None:
    schema_path = Path(schema_dir)
    data_path = Path(data_dir)
    anomalies_path = Path(anomalies_dir)
    output_data_path = Path(output_data)
    output_anomalies_path = Path(output_anomalies)
    
    schema = tfdv.load_schema_text(schema_path / "schema.pbtxt")
    csv_files = glob.glob(f"{data_path}/*.csv")
    print(csv_files)
    if len(csv_files) <= 0:
        print("No csv files found.")
        return None
    
    for file in csv_files:
        file_name = file.split("/")[-1]
        print(file_name)
        anomaly_file = f"{anomalies_path}/[{file_name}]_anomalies.pbtxt"
        anomaly_file_md = f"{anomalies_path}/[{file_name}]_anomalies_summary.md"
        print(anomaly_file)
        print(anomaly_file_md)
        anomalies = tfdv.load_anomalies_text(anomaly_file)

        if is_anomaly_present(anomalies):
            print("Anomalies found...")
            clean_df = clean_df_with_schema(data_path, file_name, schema, anomalies)
            stats = tfdv.generate_statistics_from_dataframe(clean_df)
            anomalies = tfdv.validate_statistics(stats, schema)
        
            clean_df.to_csv(f"{output_data_path}/{file_name}", index=False)
            _save_anomalies_pbtxt(output_anomalies_path, anomalies, file_name)
            _save_anomalies_to_md(output_anomalies_path, anomalies, file_name)
            print(f"Anomalies fixed. Please check the file [{file}] for more info.")
        else:
            with contextlib.suppress(FileNotFoundError):
                os.remove(anomaly_file)
                os.remove(anomaly_file_md)
            print("Empty anomaly files deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--anomalies-dir", type=str)
    parser.add_argument("--output-data-dir", type=str)
    parser.add_argument("--output-anomalies-dir", type=str)
    args = parser.parse_args()

    fix_anomalies(args.schema_dir, args.data_dir, args.anomalies_dir, args.output_data_dir, args.output_anomalies_dir)