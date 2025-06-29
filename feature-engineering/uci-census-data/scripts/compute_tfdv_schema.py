import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
import glob
import pandas as pd

def combine_csv(all_files):
    list_of_dfs = []
    
    for filename in all_files:
        single_df = pd.read_csv(filename)
        list_of_dfs.append(single_df)

    return pd.concat(list_of_dfs, ignore_index=True)

# Function to get feature by name
def get_feature_by_name(schema, name):
    for feature in schema.feature:
        if feature.name == name:
            return feature
    return None

def compute_tfdv_schema():
    from run_pipeline import CLEAN_DIR, OUTPUT_DIR

    combine_df = combine_csv(glob.glob(f"{CLEAN_DIR}/*.csv"))
    stats = tfdv.generate_statistics_from_dataframe(combine_df)
    tfdv.write_stats_text(stats, f"{OUTPUT_DIR}/baseline_stats.txt")

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

    education_num_feature = get_feature_by_name(schema, 'education.num')
    education_num_feature.ClearField('presence')
    education_num_feature.ClearField('shape')
    #education_num_feature.name = 'education.num'
    education_num_feature.type = schema_pb2.FeatureType.INT
    education_num_feature.int_domain.min = 1
    education_num_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    capital_gain_feature = get_feature_by_name(schema, 'capital.gain')
    capital_gain_feature.ClearField('presence')
    capital_gain_feature.ClearField('shape')
    #capital_gain_feature.name = 'capital.gain'
    capital_gain_feature.type = schema_pb2.FeatureType.INT
    capital_gain_feature.int_domain.min = 0
    capital_gain_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    capital_loss_feature = get_feature_by_name(schema, 'capital.loss')
    capital_loss_feature.ClearField('presence')
    capital_loss_feature.ClearField('shape')
    #capital_loss_feature.name = 'capital.loss'
    capital_loss_feature.type = schema_pb2.FeatureType.INT
    capital_loss_feature.int_domain.min = 0
    capital_loss_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    hrs_per_week_feature = get_feature_by_name(schema, 'hours.per.week')
    hrs_per_week_feature.ClearField('presence')
    hrs_per_week_feature.ClearField('shape')
    #hrs_per_week_feature.name = 'hours.per.week'
    hrs_per_week_feature.type = schema_pb2.FeatureType.INT
    hrs_per_week_feature.int_domain.min = 1
    hrs_per_week_feature.int_domain.max = 99
    hrs_per_week_feature.drift_comparator.jensen_shannon_divergence.threshold = 0.1  # 10% proportion change allowed

    # race = get_feature_by_name(schema, 'race')
    # race.distribution_constraints.min_domain_mass = 0.95  # Accept small drift

    tfdv.write_schema_text(schema, f"{OUTPUT_DIR}/schema.pbtxt")
