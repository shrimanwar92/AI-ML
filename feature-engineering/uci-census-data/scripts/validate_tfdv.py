import tensorflow as tf
print('TF: {}'.format(tf.__version__))
import numpy as np
import tensorflow_data_validation as tfdv
print('TFDV version:', tfdv.version.__version__)
from tensorflow_metadata.proto.v0 import schema_pb2, anomalies_pb2
import pandas as pd

def validate_data(new_file):
    from run_pipeline import OUTPUT_DIR

    eval_stats = tfdv.generate_statistics_from_csv(new_file)
    baseline_stats = tfdv.load_stats_text(f'{OUTPUT_DIR}/baseline_stats.txt')
    schema = tfdv.load_schema_text(f'{OUTPUT_DIR}/schema.pbtxt')

    anomalies = tfdv.validate_statistics(
        statistics=eval_stats,
        schema=schema,
        previous_statistics=baseline_stats
    )
    tfdv.write_anomalies_text(anomalies, f'{OUTPUT_DIR}/anomalies.pbtxt')

    # diaplay anomalies
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        print(f"\n=== {feature_name} ===")
        feature_schema = next((f for f in schema.feature if f.name == feature_name), None)
        for reason in anomaly_info.reason:
            short_desc = reason.short_description.lower()
            print(reason)
