import tensorflow as tf
print('TF: {}'.format(tf.__version__))
import numpy as np
import tensorflow_data_validation as tfdv
print('TFDV version:', tfdv.version.__version__)
from tensorflow_metadata.proto.v0 import schema_pb2, anomalies_pb2
from utils import OUTPUT_DIR
from typing import Tuple, List
from zenml import step
from pathlib import Path

def save_anomalies_to_md(anomalies, stem):
    anomalies_md_path = f'{OUTPUT_DIR}/[{stem}]_anomalies_summary.md'
    with open(anomalies_md_path, 'w') as f:
        f.write(f"# Data Anomalies Detected in file {stem}\n\n")
        for feature, details in anomalies.anomaly_info.items():
            print(feature)
            f.write(f"### Feature: `{feature}`\n")
            f.write(f"- Description: {details.description}\n")
            f.write(f"- Severity: {details.severity}\n\n")
            for reason in details.reason:
                f.write(f"- Reason: {reason}\n\n")
                print(reason)

def save_anomalies_pbtxt(anomalies, stem):
    anomalies_path = f'{OUTPUT_DIR}/[{stem}]_anomalies.pbtxt'   
    tfdv.write_anomalies_text(anomalies, anomalies_path)


def merge_anomalies(anomalies_list):
    combined_anomalies = anomalies_pb2.Anomalies()

    for anomalies in anomalies_list:
        for feature, info in anomalies.anomaly_info.items():
            if feature not in combined_anomalies.anomaly_info:
                combined_anomalies.anomaly_info[feature].CopyFrom(info)
            else:
                # Merge descriptions/severity/reasons intelligently
                existing_info = combined_anomalies.anomaly_info[feature]
                if info.description not in existing_info.description:
                    existing_info.description += f"; {info.description}"
                existing_info.severity = max(existing_info.severity, info.severity)
                existing_info.reason.extend(info.reason)

    return combined_anomalies

@step(enable_cache=False)
def validate_csv(cleaned_csv: List[str], schema_path: str, stats_path: str) -> bool:
    print("Validating csv..")
    schema = tfdv.load_schema_text(schema_path)
    baseline_stats = tfdv.load_statistics(stats_path)
    all_anomalies = []

    for file in cleaned_csv:
        new_stats = tfdv.generate_statistics_from_csv(data_location=file)
        anomalies = tfdv.validate_statistics(
            statistics=new_stats,
            schema=schema,
            previous_statistics=baseline_stats
        )
        anomalies = tfdv.validate_statistics(
            statistics=new_stats,
            schema=schema,
            previous_statistics=baseline_stats
        )
        stem = file.split("/")[-1]
        save_anomalies_pbtxt(anomalies, stem)
        save_anomalies_to_md(anomalies, stem)

    return True
