import os
import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from helpers import DEFAULT_REGION, S3_BUCKET, get_pandas_processor, get_tensorflow_processor

# Dummy AWS credentials
os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION

def _preprocess(script_processor):
    # ✅ Run preprocess script
    pandas_processor = get_pandas_processor()

    pandas_processor.run(
        code="steps/preprocess.py",
        inputs=[
            ProcessingInput(source=os.path.abspath("dataset/raw"), destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{S3_BUCKET}/dataset/cleaned"),
        ],
        arguments=["--input-dir", "/opt/ml/processing/input", "--output-dir", "/opt/ml/processing/output"],
        logs=True
    )

def _compute_schema():
    script_processor = get_tensorflow_processor("compute_schema")
    
    # Step 2: Compute Schema
    script_processor.run(
        code="steps/compute_tfdv_schema.py",
        inputs=[
            ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/cleaned", destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{S3_BUCKET}/output")
        ],
        arguments=["--input-dir", "/opt/ml/processing/input", "--output-dir", "/opt/ml/processing/output"],
    )

def _validate_csv():
    script_processor = get_tensorflow_processor("validate_csv")
    
    script_processor.run(
        code="steps/tfdv_validate.py",
        inputs=[
            ProcessingInput(source=f"s3://{S3_BUCKET}/output", destination="/opt/ml/processing/schema"),
            ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/cleaned", destination="/opt/ml/processing/data")
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{S3_BUCKET}/anomalies")
        ],
        arguments=[
            "--schema-dir", "/opt/ml/processing/schema", 
            "--data-dir", "/opt/ml/processing/data",
            "--output-dir", "/opt/ml/processing/output"
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
            ProcessingInput(source=f"s3://{S3_BUCKET}/output", destination=schema_dir),
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


def run_pipeline():
    _preprocess()
    #_compute_schema()
    #_validate_csv()
    #_fix_anomalies()
    #_transform(analyze=False)

if __name__ == "__main__":
    run_pipeline()
