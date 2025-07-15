import os
import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from helpers import setup_sagemaker_local_session, DEFAULT_REGION, ROLE, S3_BUCKET

# Dummy AWS credentials
os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION

def _preprocess(script_processor):
    # ✅ Run preprocess script
    script_processor.run(
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

def _compute_schema(script_processor):
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

def _validate_csv(script_processor):
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

def _fix_anomalies(script_processor):
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


def run_pipeline():
    local_session = setup_sagemaker_local_session()

    # ✅ ScriptProcessor with your custom Docker image
    script_processor = ScriptProcessor(
        image_uri="sagemaker-local:latest",
        command=["python3"],
        instance_type="local",
        instance_count=1,
        base_job_name="preprocess",
        role=ROLE,
        sagemaker_session=local_session,
    )

    #_preprocess(script_processor)
    #_compute_schema(script_processor)
    #_validate_csv(script_processor)
    _fix_anomalies(script_processor)

if __name__ == "__main__":
    run_pipeline()
