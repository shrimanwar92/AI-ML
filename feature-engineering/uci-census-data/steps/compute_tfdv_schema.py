import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
from typing import Tuple, List
from zenml import step
from utils import combine_csv, OUTPUT_DIR, get_feature_by_name

@step(enable_cache=False)
def compute_tfdv_schema(cleaned_csvs: List[str])  -> Tuple[str, str]:
    stats_path = f"{OUTPUT_DIR}/baseline_stats.txt"
    schema_path = f"{OUTPUT_DIR}/schema.pbtxt"
    
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

    fnlwgt_feature = get_feature_by_name(schema, 'fnlwgt')
    fnlwgt_feature.ClearField('presence')
    fnlwgt_feature.ClearField('shape')
    fnlwgt_feature.type = schema_pb2.FeatureType.INT
    
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
    