import site
site.addsitedir("/home/nilays/tfx/lib/python3.11/site-packages")

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
from utils import OUTPUT_DIR, DATASET_DIR, combine_csv
from typing import List, Dict
from pathlib import Path
import shutil


# --- Step 1: Define preprocessing function ---
def preprocessing_fn(inputs):
    outputs = {}
    numeric = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    for f in numeric:
        mean = tft.mean(inputs[f])
        var = tft.var(inputs[f])
        tf.compat.v1.logging.info(f'Mean and Var for {f}: %s, %s', mean, var)
        outputs[f"{f}_normalized"] = tft.scale_to_z_score(inputs[f])
    
    categorical = ['workclass', 'education', 'marital_status', 'occupation', 
                   'relationship', 'race', 'sex', 'native_country']
    for f in categorical:
        outputs[f"{f}_encoded"] = tft.compute_and_apply_vocabulary(inputs[f])
    
    outputs['label'] = tf.cast(tft.compute_and_apply_vocabulary(inputs['income']), tf.int64)
    
    return outputs


# # --- Step 2: Define metadata (raw schema) ---
RAW_FEATURE_SPEC = {
    'age': tf.io.FixedLenFeature([], tf.float32),
    'fnlwgt': tf.io.FixedLenFeature([], tf.float32),
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
}

RAW_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(RAW_FEATURE_SPEC)
)

# --- Step 4: Run Beam pipeline ---
SAVE_DIR = "tmp/transform_output"
TRANSFORM_FN_DIR = OUTPUT_DIR / "transform_fn"
PARQUET_OUTPUT = DATASET_DIR / "transformed"
os.makedirs(SAVE_DIR, exist_ok=True)
if os.path.exists(TRANSFORM_FN_DIR):
    shutil.rmtree(TRANSFORM_FN_DIR)
os.makedirs(TRANSFORM_FN_DIR, exist_ok=True)
os.makedirs(PARQUET_OUTPUT, exist_ok=True)

def _load_data(file_paths: List[str]):
    # --- Step 3: Load raw data (example CSV) ---
    # '/mnt/c/Users/shrim/Documents/src/AI-ML/feature-engineering/uci-census-data/data.csv'
    #df = pd.read_csv(file_path)
    df = combine_csv(file_paths)
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
                _ = transform_fn | "WriteTransformFn" >> tft_beam.WriteTransformFn(TRANSFORM_FN_DIR)

            else:
                # Load existing transform_fn and only transform
                transform_fn = (
                    pipeline
                    | "ReadTransformFn" >> tft_beam.ReadTransformFn(TRANSFORM_FN_DIR)
                )

                transformed_data_pc, transformed_metadata = (
                    ((raw_data_pc, RAW_METADATA), transform_fn)
                    | "TransformOnly" >> tft_beam.TransformDataset()
                )

            # Write output as a single parquet file using Arrow
            def to_arrow_table(data_iter):
                df = pd.DataFrame(data_iter)
                print("Sample transformed rows:\n", df.head())
                
                table = pa.Table.from_pandas(df)
                pq.write_table(table, os.path.join(PARQUET_OUTPUT, f"{output_file_name}.parquet"))
                return []

            _ = (
                transformed_data_pc
                | "ToList" >> beam.combiners.ToList()
                | "WriteParquet" >> beam.FlatMap(to_arrow_table)
            )


def transform_data(file_paths: List[str], analyze: bool) -> None:
    output_file_name = "transformed_"
    for file_name in file_paths:
        output_file_name += f"{Path(file_name).stem}_"
    
    raw_data = _load_data(file_paths)
    
    run_pipeline(
        raw_data=raw_data,
        analyze=analyze,
        output_file_name=output_file_name
    )