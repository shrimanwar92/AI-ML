import os
from sagemaker.processing import ProcessingInput, ProcessingOutput
from helpers import (DEFAULT_REGION, S3_BUCKET, 
                    get_pandas_processor, get_tensorflow_processor,
                    local_session, ROLE)
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline

# Dummy AWS credentials
os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION


def _compute_schema():
    script_processor = get_tensorflow_processor("compute_schema")
    container_input_dir = "/opt/ml/processing/input"
    container_output_dir = "/opt/ml/processing/output"

    return ProcessingStep(
        name="ComputeSchemaAndBaselineStats",
        processor=script_processor,
        inputs=[
            ProcessingInput(source=os.path.abspath("dataset/baseline"), destination=container_input_dir),
        ],
        outputs=[
            ProcessingOutput(source=container_output_dir, destination=f"s3://{S3_BUCKET}/schema")
        ],
        code="scripts/compute_schema.py",
        job_arguments=["--baseline-input-dir", container_input_dir , "--output-dir", container_output_dir],
    )

def _preprocess():
    # ✅ Run preprocess script
    pandas_processor = get_pandas_processor()
    container_input_dir = "/opt/ml/processing/input"
    container_output_dir = "/opt/ml/processing/output"

    return ProcessingStep(
        name="PreprocessData",
        processor=pandas_processor,
        inputs=[
            ProcessingInput(source=os.path.abspath("dataset/raw"), destination=container_input_dir),
        ],
        outputs=[
            ProcessingOutput(source=container_output_dir, destination=f"s3://{S3_BUCKET}/dataset/cleaned"),
        ],
        code="scripts/preprocess.py",
        job_arguments=["--input-dir", container_input_dir, "--output-dir", container_output_dir]
    )


def _validate_csv():
    script_processor = get_tensorflow_processor("validate_csv")
    container_data_dir = "/opt/ml/processing/data"
    container_output_dir = "/opt/ml/processing/output"
    container_schema_dir = "/opt/ml/processing/schema"
    
    script_processor.run(
        code="steps/tfdv_validate.py",
        inputs=[
            ProcessingInput(source=f"s3://{S3_BUCKET}/schema", destination=container_schema_dir),
            ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/cleaned", destination=container_data_dir)
        ],
        outputs=[
            ProcessingOutput(source=container_output_dir, destination=f"s3://{S3_BUCKET}/anomalies")
        ],
        arguments=[
            "--schema-dir", container_schema_dir, 
            "--data-dir", container_data_dir,
            "--output-dir", container_output_dir
        ],
    )

def _fix_anomalies():
    script_processor = get_tensorflow_processor("fix_anomalies")
    
    schema_dir = "/opt/ml/processing/schema"
    data_dir = "/opt/ml/processing/data"
    anomalies_dir = "/opt/ml/processing/anomalies"
    output_data_dir = "/opt/ml/processing/data/output"
    output_anomalies_dir = "/opt/ml/processing/anomalies/output"
    
    
    script_processor.run(
        code="steps/fix_anomaly.py",
        inputs=[
            ProcessingInput(source=f"s3://{S3_BUCKET}/schema", destination=schema_dir),
            ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/cleaned", destination=data_dir),
            ProcessingInput(source=f"s3://{S3_BUCKET}/anomalies", destination=anomalies_dir),
            ProcessingInput(source="steps", destination="/opt/ml/code"),
        ],
        outputs=[
            ProcessingOutput(source=output_data_dir, destination=f"s3://{S3_BUCKET}/dataset/cleaned"),
            ProcessingOutput(source=output_anomalies_dir, destination=f"s3://{S3_BUCKET}/anomalies"),
        ],
        arguments=[
            "--schema-dir", schema_dir, 
            "--data-dir", data_dir,
            "--anomalies-dir", anomalies_dir,
            "--output-data-dir", output_data_dir,
            "--output-anomalies-dir", output_anomalies_dir
        ],
    )

def _transform(analyze: bool):
    script_processor = get_tensorflow_processor("transform")
    
    if analyze:
        arguments = [
            "--files", "adult.csv", "adult_drifted_invalid.csv",
            "--analyze", "true",
            "--data-dir", "/opt/ml/processing/data",
            "--transformed-output-dir", "/opt/ml/processing/output/transformed",
            "--transform-fn-output-dir", "/opt/ml/processing/output/transform_fn"
        ]
        inputs = [
            ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/cleaned", destination="/opt/ml/processing/data")
        ]
        outputs = [
            ProcessingOutput(source="/opt/ml/processing/output/transformed", destination=f"s3://{S3_BUCKET}/dataset/transformed"),
            ProcessingOutput(source="/opt/ml/processing/output/transform_fn", destination=f"s3://{S3_BUCKET}/output/transform_fn")
        ]
    else:
        arguments = [
            "--files", "SRmopfnncL.csv",
            "--analyze", "false",
            "--data-dir", "/opt/ml/processing/data",
            "--transform-fn-dir", "/opt/ml/processing/transform_fn",
            "--transformed-output-dir", "/opt/ml/processing/output/transformed"
        ]
        inputs = [
            ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/cleaned", destination="/opt/ml/processing/data"),
            ProcessingInput(source=f"s3://{S3_BUCKET}/output/transform_fn", destination="/opt/ml/processing/transform_fn")
        ]
        outputs = [
            ProcessingOutput(source="/opt/ml/processing/output/transformed", destination=f"s3://{S3_BUCKET}/dataset/transformed"),
        ]

    script_processor.run(
        code="steps/transform.py",
        inputs=inputs,
        outputs=outputs,
        arguments=arguments,
    )

def _store_features_in_feature_store():
    script_processor = get_pandas_processor("store_features")
    container_data_dir = "/opt/ml/processing/data"
    
    script_processor.run(
        code="steps/store_features_in_feature_store.py",
        inputs=[
            ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/transformed", destination=container_data_dir)
        ],
        arguments=[
            "--data-dir", container_data_dir,
            "--role-arn", "arn:aws:iam::229058914239:role/DummyRole",
            "--files", "transformed_adult_adult_drifted_invalid__features.parquet",
        ],
    )

pipeline = Pipeline(
    name="FeatureEnggPipeline",
    steps=[_preprocess()],
    sagemaker_session=local_session
)

pipeline.upsert(role_arn=ROLE)
execution = pipeline.start()


# def run_pipeline():
#     # _preprocess()
#     _compute_schema()
#     # _validate_csv()
#     #_fix_anomalies()
#     #_transform(analyze=False)
#     #_store_features_in_feature_store()


# if __name__ == "__main__":
#     run_pipeline()
