import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
from typing import Tuple
import argparse
from pathlib import Path
import glob

def get_feature_by_name(schema, name):
    for feature in schema.feature:
        if feature.name == name:
            return feature
    return None

def combine_csv(all_files):
    import pandas as pd
    
    list_of_dfs = []
    
    for filename in all_files:
        single_df = pd.read_csv(filename)
        list_of_dfs.append(single_df)

    return pd.concat(list_of_dfs, ignore_index=True)

def compute_tfdv_schema(raw_input_dir, clean_output_dir)  -> Tuple[str, str]:
    raw_input_path = Path(raw_input_dir)
    cleaned_csvs = glob.glob(f"{raw_input_path}/*.csv")
    clean_output_path = Path(clean_output_dir)
    clean_output_path.mkdir(parents=True, exist_ok=True)
    
    stats_path = clean_output_path / "baseline_stats.txt"
    schema_path = clean_output_path / "schema.pbtxt"
    
    combine_df = combine_csv(cleaned_csvs)
    stats = tfdv.generate_statistics_from_dataframe(combine_df)

    schema = tfdv.infer_schema(stats)
    #schema = schema_pb2.Schema()
    
    for feature in schema.feature:
        if feature.type == schema_pb2.FeatureType.BYTES:
            feature.ClearField('presence')
            feature.ClearField('shape')

    age_feature = get_feature_by_name(schema, 'age')
    age_feature.ClearField('presence')
    age_feature.ClearField('shape')
    #age_feature.name = 'age'
    age_feature.type = schema_pb2.FeatureType.INT
    age_feature.int_domain.min = 17
    age_feature.int_domain.max = 90
    age_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed
    
    education_num_feature = get_feature_by_name(schema, 'education_num')
    education_num_feature.ClearField('presence')
    education_num_feature.ClearField('shape')
    #education_num_feature.name = 'education.num'
    education_num_feature.type = schema_pb2.FeatureType.INT
    education_num_feature.int_domain.min = 1
    education_num_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    capital_gain_feature = get_feature_by_name(schema, 'capital_gain')
    capital_gain_feature.ClearField('presence')
    capital_gain_feature.ClearField('shape')
    #capital_gain_feature.name = 'capital.gain'
    capital_gain_feature.type = schema_pb2.FeatureType.INT
    capital_gain_feature.int_domain.min = 0
    capital_gain_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    capital_loss_feature = get_feature_by_name(schema, 'capital_loss')
    capital_loss_feature.ClearField('presence')
    capital_loss_feature.ClearField('shape')
    #capital_loss_feature.name = 'capital.loss'
    capital_loss_feature.type = schema_pb2.FeatureType.INT
    capital_loss_feature.int_domain.min = 0
    capital_loss_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    hrs_per_week_feature = get_feature_by_name(schema, 'hours_per_week')
    hrs_per_week_feature.ClearField('presence')
    hrs_per_week_feature.ClearField('shape')
    #hrs_per_week_feature.name = 'hours.per.week'
    hrs_per_week_feature.type = schema_pb2.FeatureType.INT
    hrs_per_week_feature.int_domain.min = 1
    hrs_per_week_feature.int_domain.max = 99
    hrs_per_week_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    race = get_feature_by_name(schema, 'race')
    race.distribution_constraints.min_domain_mass = 0.95  # Accept small drift
    
    tfdv.write_stats_text(stats, stats_path)
    tfdv.write_schema_text(schema, schema_path)

    return schema_path, stats_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    compute_tfdv_schema(args.input_dir, args.output_dir)
    