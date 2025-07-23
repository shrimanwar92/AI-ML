import os
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import apache_beam as beam
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from typing import List, Dict
from pathlib import Path
import argparse
from pathlib import Path

TRANSFORM_FN_OUTPUT_DIR = None
PARQUET_OUTPUT = None
TRANSFORM_FN_READ_DIR = None

def _combine_csv(all_files):
    import pandas as pd
    
    list_of_dfs = []
    
    for filename in all_files:
        single_df = pd.read_csv(filename)
        list_of_dfs.append(single_df)

    return pd.concat(list_of_dfs, ignore_index=True)


# --- Step 1: Define preprocessing function ---
def preprocessing_fn(inputs):
    outputs = {}
    numeric = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    for f in numeric:
        mean = tft.mean(inputs[f])
        var = tft.var(inputs[f])
        tf.compat.v1.logging.info(f'Mean and Var for {f}: %s, %s', mean, var)
        outputs[f"{f}_normalized"] = tft.scale_to_z_score(inputs[f])
    
    categorical = ['workclass', 'education', 'marital_status', 'occupation', 
                   'relationship', 'race', 'sex', 'native_country']
    for f in categorical:
        outputs[f"{f}_encoded"] = tft.compute_and_apply_vocabulary(inputs[f])

    income = inputs['income']
    cleaned_income = tf.where(
        tf.logical_or(tf.equal(income, "<=50K"), tf.equal(income, ">50K")),
        income,
        tf.constant("<=50K")  # Replace invalid with default or estimated most common
    )
    
    outputs['label'] = tf.cast(tft.compute_and_apply_vocabulary(cleaned_income), tf.int64)

    outputs['record_id'] = inputs['record_id']
    outputs['event_time'] = inputs['event_time']
    
    return outputs


# # --- Step 2: Define metadata (raw schema) ---
RAW_FEATURE_SPEC = {
    'record_id': tf.io.FixedLenFeature([], tf.string),
    'age': tf.io.FixedLenFeature([], tf.float32),
    'education_num': tf.io.FixedLenFeature([], tf.float32),
    'capital_gain': tf.io.FixedLenFeature([], tf.float32),
    'capital_loss': tf.io.FixedLenFeature([], tf.float32),
    'hours_per_week': tf.io.FixedLenFeature([], tf.float32),
    'workclass': tf.io.FixedLenFeature([], tf.string),
    'education': tf.io.FixedLenFeature([], tf.string),
    'marital_status': tf.io.FixedLenFeature([], tf.string),
    'occupation': tf.io.FixedLenFeature([], tf.string),
    'relationship': tf.io.FixedLenFeature([], tf.string),
    'race': tf.io.FixedLenFeature([], tf.string),
    'sex': tf.io.FixedLenFeature([], tf.string),
    'native_country': tf.io.FixedLenFeature([], tf.string),
    'income': tf.io.FixedLenFeature([], tf.string),
    'event_time': tf.io.FixedLenFeature([], tf.string),
}

RAW_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(RAW_FEATURE_SPEC)
)

def _load_data(file_paths: List[str]):
    df = _combine_csv(file_paths)
    raw_data = df.to_dict(orient="records")
    return raw_data

def run_pipeline(raw_data: List[Dict], analyze: bool, output_file_name: str):
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir="tmp"):
            raw_data_pc = pipeline | "CreateRawData" >> beam.Create(raw_data)

            if analyze:
                # Analyze and transform in the same run
                (transformed_data_pc, transformed_metadata), transform_fn = (
                    (raw_data_pc, RAW_METADATA)
                    | "AnalyzeAndTransform" >> tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
                )

                # Save transform_fn for reuse
                _ = transform_fn | "WriteTransformFn" >> tft_beam.WriteTransformFn(TRANSFORM_FN_OUTPUT_DIR)

            else:
                # Load existing transform_fn and only transform
                transform_fn = (
                    pipeline
                    | "ReadTransformFn" >> tft_beam.ReadTransformFn(TRANSFORM_FN_READ_DIR)
                )

                transformed_data_pc, transformed_metadata = (
                    ((raw_data_pc, RAW_METADATA), transform_fn)
                    | "TransformOnly" >> tft_beam.TransformDataset()
                )

            # Write output as a single parquet file using Arrow
            def to_arrow_table(data_iter):
                df = pd.DataFrame(data_iter)

                print("Sample transformed rows:\n", df.head())

                # Separate label, id, and event time
                label_df = df[["label", "record_id", "event_time"]]
                feature_columns = [col for col in df.columns if col not in ["label", "record_id", "event_time"]]
                feature_df = df[["record_id", "event_time"] + feature_columns]

                # Write label data for model training
                pq.write_table(pa.Table.from_pandas(label_df), os.path.join(PARQUET_OUTPUT, f"{output_file_name}_labels.parquet"))

                # Write features for Feature Store ingestion
                pq.write_table(pa.Table.from_pandas(feature_df), os.path.join(PARQUET_OUTPUT, f"{output_file_name}_features.parquet"))

                return []


            _ = (
                transformed_data_pc
                | "ToList" >> beam.combiners.ToList()
                | "WriteParquet" >> beam.FlatMap(to_arrow_table)
            )


def _transform_dataset(file_paths: List[str], analyze: bool) -> None:
    output_file_name = "transformed_"
    for file_name in file_paths:
        output_file_name += f"{Path(file_name).stem}_"
    
    raw_data = _load_data(file_paths)
    
    run_pipeline(
        raw_data=raw_data,
        analyze=analyze,
        output_file_name=output_file_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', help="List of files to process")
    parser.add_argument("--analyze", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--transformed-output-dir", type=str)
    parser.add_argument("--transform-fn-output-dir", type=str, default="/opt/ml/processing/output/transform_fn")
    parser.add_argument("--transform-fn-dir", type=str, default="/opt/ml/processing/transform_fn")
    args = parser.parse_args()

    PARQUET_OUTPUT = Path(args.transformed_output_dir)
    file_paths = [f"{args.data_dir}/{file}" for file in args.files]

    if args.analyze.lower() == "true":
        TRANSFORM_FN_OUTPUT_DIR = Path(args.transform_fn_output_dir)
        _transform_dataset(file_paths, True)
    else:
        TRANSFORM_FN_READ_DIR = Path(args.transform_fn_dir)
        _transform_dataset(file_paths, False)