from zenml import step
import tensorflow_data_validation as tfdv
from utils import OUTPUT_DIR, CLEAN_DIR
import glob
import pandas as pd
from steps.tfdv_validate import save_anomalies_to_md, save_anomalies_pbtxt
from typing import List
import contextlib
import os

# Function to get feature by name
def get_feature_by_name(schema, name):
    for feature in schema.feature:
        if feature.name == name:
            return feature
    return None

def clean_df_with_schema(file_name, schema, anomalies):
    df = pd.read_csv(f'{CLEAN_DIR}/{file_name}')

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


@step(enable_cache=False)
def fix_anomalies(csv_files: List[str]) -> None:
    """Check OUTPUT_DIR for anomalies.pbtxt files, fix them, and return updated schema/stats paths."""
    
    schema_path = f"{OUTPUT_DIR}/schema.pbtxt"
    
    if len(csv_files) <= 0:
        print("No anomaly files found in OUTPUT_DIR")
        return None
    
    for file in csv_files:
        file_name = file.split("/")[-1]
        print(file_name)
        anomaly_file = f"{OUTPUT_DIR}/[{file_name}]_anomalies.pbtxt"
        anomaly_file_md = f"{OUTPUT_DIR}/[{file_name}]_anomalies_summary.md"
        print(anomaly_file)
        print(anomaly_file_md)
        anomalies = tfdv.load_anomalies_text(anomaly_file)
        schema = tfdv.load_schema_text(schema_path)

        if len(anomalies.anomaly_info) > 0:
            print("Anomalies found...")
            clean_df = clean_df_with_schema(file_name, schema, anomalies)
            stats = tfdv.generate_statistics_from_dataframe(clean_df)
            anomalies = tfdv.validate_statistics(stats, schema)
        
            print("Trying to fix anomalies...")
            clean_df.to_csv(f"{CLEAN_DIR}/{file_name}", index=False)
            save_anomalies_pbtxt(anomalies, file_name)
            save_anomalies_to_md(anomalies, file_name)
            print(f"Anomalies fixed. Please check the file [{file}] for more info.")
        else:
            with contextlib.suppress(FileNotFoundError):
                os.remove(anomaly_file)
                os.remove(anomaly_file_md)
            print("Empty anomaly files deleted.")