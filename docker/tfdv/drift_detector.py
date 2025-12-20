import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2
import pandas as pd
import os
from typing import Dict, Any

print("TFDV version: ", tfdv.__version__)

# Define a central directory for artifacts (Schema, Stats, Anomalies)
# This path is relative to the container's WORKDIR (/app)
ARTIFACTS_DIR = 'data_drift_artifacts' 
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PROJECT_NAME = 'ecommerce_product'
schema_path = os.path.join(ARTIFACTS_DIR, f'{PROJECT_NAME}_schema.pbtxt')

# --- 1. Utility Functions (Same as before) ---

def load_schema(schema_path):
    """Loads the schema with pre-defined drift thresholds."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Schema file not found at {schema_path}. RUN INIT PHASE FIRST."
        )
    print(f"Loading schema: {schema_path}")
    return tfdv.load_schema_text(schema_path)

def get_stats(df, name):    
    # Always regenerate for a fresh run, but in MLOps you would load baseline stats
    print(f"Generating {name} statistics...")
    stats = tfdv.generate_statistics_from_dataframe(df)
    
    # with open(stats_path, 'wb') as f:
    #     f.write(stats.SerializeToString())
    # print(f"{name} stats saved to {stats_path}")
    
    return stats

def check_for_data_drift(
    baseline_stats,
    production_stats,
    schema
):
    """Compares production statistics to baseline statistics using the schema."""
    print("\n--- Running TFDV Drift Validation ---")
    
    anomalies = tfdv.validate_statistics(
        statistics=production_stats,
        schema=schema,
        previous_statistics=baseline_stats  # Key argument for drift detection
    )
    
    tfdv.display_anomalies(anomalies) # Display TFDV report in terminal logs
    
    return anomalies

def analyze_drift_results(anomalies, name):
    """Parses the Anomalies object for status."""
    drift_status = "NO_DRIFT"
    drifted_features = []

    print(anomalies)
    anomalies_path = os.path.join(ARTIFACTS_DIR, f'{PROJECT_NAME}_{name}_stats.pbtxt')
    tfdv.write_anomalies_text(anomalies, anomalies_path)
    
    # for feature_name, anomaly_info in anomalies.anomaly_info.items():
    #     if anomaly_info.short_description == 'Distribution change':
    #         drifted_features.append(feature_name)
    #         drift_status = "MAJOR_DRIFT"
    #     elif anomaly_info.severity == anomalies_pb2.AnomalyInfo.Severity.ERROR:
    #         drift_status = "SCHEMA_ERROR"
    #         break 
    
    # return {'status': drift_status, 'drifted_features': drifted_features}

# --- 2. Data Initialization and Schema Creation (Run ONLY ONCE for MLOps project start) ---

def initialize_project_artifacts(schema_path, df_baseline):
    """Initial phase to create the schema and set drift thresholds."""
    if os.path.exists(schema_path):
        print("Schema already exists. Skipping initialization.")
        return

    print("--- Initializing Project Artifacts (First Run) ---")
    baseline_stats = get_stats(df_baseline, 'baseline_temp')
    schema = tfdv.infer_schema(baseline_stats)
    
    # 1. Define drift thresholds directly on the schema object
    # The 'category' feature is expected to be stable. Set a strict L-inf threshold of 0.05 (5% change).
    tfdv.get_feature(schema, 'category').drift_comparator.infinity_norm.threshold = 0.05
    
    # 2. Save the schema (This is the configuration file for the MLOps team)
    tfdv.write_schema_text(schema, schema_path)
    print(f"Saved initial schema with drift thresholds to: {schema_path}")
    
    
# --- 3. Main Execution ---

if __name__ == '__main__':    
    # 1. BASELINE DATA (Training Data)
    df_baseline = pd.DataFrame({
        # Baseline: High volume of 'Electronics' and 'Clothing'
        'product_id': range(1000),
        'category': ['Electronics'] * 500 + ['Clothing'] * 400 + ['Books'] * 100,
        'price': [150.0] * 1000
    })

    # 2. PRODUCTION DATA (Serving Data - SIMULATING DRIFT)
    df_production = pd.DataFrame({
        'product_id': range(1000),
        # Drift: A new category 'Home Goods' suddenly dominates, while 'Electronics' drops significantly.
        'category': ['Home Goods'] * 600 + ['Clothing'] * 300 + ['Electronics'] * 100,
        'price': [155.0] * 1000 # Minor price change, unlikely to trigger a high threshold
    })

    initialize_project_artifacts(schema_path, df_baseline)
    final_schema = load_schema(schema_path)
    baseline_stats = get_stats(df_baseline, 'baseline')
    production_stats = get_stats(df_production, 'production')
    anomalies = check_for_data_drift(baseline_stats, production_stats, final_schema)
    analysis_results = analyze_drift_results(anomalies, "production")

    # --- Report ---
    # print("\n==================================")
    # print(f"PROJECT: {PROJECT_NAME} Drift Check")
    # print(f"STATUS: {analysis_results['status']}")
    # if analysis_results['drifted_features']:
    #     print(f"CRITICAL DRIFT DETECTED in: {', '.join(analysis_results['drifted_features'])}")
    #     print("ACTION: Investigate new 'Home Goods' category distribution.")
    # print("==================================")