from zenml import step
import tensorflow_data_validation as tfdv
from utils import OUTPUT_DIR, CLEAN_DIR
import glob
import pandas as pd
from steps.tfdv_validate import save_anomalies_to_md, save_anomalies_pbtxt

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
            print(f">>>>>>>>{feature_name}>>>>>>>>>>", short_desc)
            if "out-of-range" in short_desc and (feature_schema and feature_schema.int_domain):
                min_expected = feature_schema.int_domain.min
                max_expected = feature_schema.int_domain.max
                print(f"🧹 Clipping [{feature_name}] to [{min_expected}, {max_expected}]")
                df.loc[df[feature_name] < min_expected, feature_name] = min_expected
                df.loc[df[feature_name] > max_expected, feature_name] = max_expected

    return df


@step
def fix_anomalies() -> None:
    """Check OUTPUT_DIR for anomalies.pbtxt files, fix them, and return updated schema/stats paths."""
    anomaly_files = glob.glob(f"{OUTPUT_DIR}/*anomalies.pbtxt")
    schema_path = f"{OUTPUT_DIR}/schema.pbtxt"
    stats_path = f"{OUTPUT_DIR}/baseline_stats.txt"

    print(anomaly_files)
    
    if not anomaly_files:
        print("No anomaly files found in OUTPUT_DIR")
        return None
    
    for file in anomaly_files:
        anomalies = tfdv.load_anomalies_text(file)
        schema = tfdv.load_schema_text(schema_path)
        csv_name = file.split("[")[1].split("]")[0]
        clean_df = clean_df_with_schema(csv_name, schema, anomalies)
        
        stats = tfdv.generate_statistics_from_dataframe(clean_df)
        anomalies = tfdv.validate_statistics(stats, schema)
        
        clean_df.to_csv(f"{CLEAN_DIR}/{csv_name}", index=False)
        save_anomalies_pbtxt(anomalies, csv_name)
        save_anomalies_to_md(anomalies, csv_name)
    
    print("Anomalies fixed.")