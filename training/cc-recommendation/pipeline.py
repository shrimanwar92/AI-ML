import os
from sagemaker.processing import ProcessingInput, ProcessingOutput
from helpers import (DEFAULT_REGION, S3_BUCKET, 
                    get_pandas_processor, get_tensorflow_processor,
                    local_session, ROLE)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet

# Dummy AWS credentials
os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION


def _tfdv_analyze(data_input_dir, create_schema):
    script_processor = get_tensorflow_processor("tfdv_validate")
    container_input_dir = "/opt/ml/processing/input"
    container_output_dir = "/opt/ml/processing/output"
    schema_dir = "/opt/ml/processing/schema"

    if create_schema == "true":
        destination=f"s3://{S3_BUCKET}/tfdv-outputs"
        inputs=[ProcessingInput(source=data_input_dir, destination=container_input_dir)]
    else:
        destination=f"s3://{S3_BUCKET}/anomalies"
        inputs=[
            ProcessingInput(source=data_input_dir, destination=container_input_dir),
            ProcessingInput(source=f"s3://{S3_BUCKET}/tfdv-outputs", destination=schema_dir),
        ]

    return ProcessingStep(
        name="Tfdv Analyze",
        processor=script_processor,
        inputs=inputs,
        outputs=[
            ProcessingOutput(source=container_output_dir, destination=destination)
        ],
        code="scripts/tfdv_analyze.py",
        job_arguments=["--input-dir", container_input_dir , 
                        "--output-dir", container_output_dir, 
                        "--create-schema", create_schema,
                        "--schema-dir", schema_dir
                    ],
    )

def _fix_anomalies(data_input_dir):
    tf_processor = get_tensorflow_processor("fix_anomalies")
    container_input_dir = "/opt/ml/processing/input"
    container_output_dir = "/opt/ml/processing/output"
    schema = ("/opt/ml/processing/schema", f"s3://{S3_BUCKET}/tfdv-outputs")
    anomalies = ("/opt/ml/processing/anomalies", f"s3://{S3_BUCKET}/anomalies")
    
    inputs=[
        ProcessingInput(source=data_input_dir, destination=container_input_dir),
        ProcessingInput(source=schema[1], destination=schema[0]),
        ProcessingInput(source=anomalies[1], destination=anomalies[0]),
    ]

    validate_fix_step = ProcessingStep(
        name="TFDVValidateFix",
        processor=tf_processor,
        inputs=inputs,
        outputs=[
            #ProcessingOutput(output_name="cleaned_intermediate", source=container_output_dir),
            ProcessingOutput(source=anomalies[0], destination=anomalies[1]),
            ProcessingOutput(source=container_output_dir, destination=f"s3://{S3_BUCKET}/dataset/cleaned")
        ],
        code="scripts/fix_anomaly.py",
        job_arguments=["--input-dir", container_input_dir, 
                        "--output-dir", container_output_dir,
                        "--schema-dir", schema[0],
                        "--anomalies-dir", anomalies[0]
                    ],
        #property_files=[anomaly_property]
    )

    return validate_fix_step

def _convert_to_parquet(input_location, output_location):
    pandas_processor = get_pandas_processor("convert_to_parquet")
    inputs = ("/opt/ml/processing/input", input_location)
    outputs = ("/opt/ml/processing/output", output_location)
    
    return ProcessingStep(
        name="ConvertToParquet",
        processor=pandas_processor,
        inputs=[
            ProcessingInput(source=inputs[1], destination=inputs[0])
        ],
        outputs=[
            ProcessingOutput(source=outputs[0], destination=outputs[1]),
            
        ],
        code="scripts/save_to_parquet.py",
        job_arguments=["--input-dir", inputs[0], 
                        "--output-dir", outputs[0]
                    ],
    )

def _train(parquet_input):
    pandas_processor = get_pandas_processor("train")
    inputs = ("/opt/ml/processing/input", parquet_input)
    outputs = ("/opt/ml/processing/output", f"s3://{S3_BUCKET}/artifacts")

    return ProcessingStep(
        name="TrainModel",
        processor=pandas_processor,
        inputs=[
            ProcessingInput(source=inputs[1], destination=inputs[0])
        ],
        outputs=[
            ProcessingOutput(source=outputs[0], destination=outputs[1]),
            
        ],
        code="scripts/train.py",
        job_arguments=["--input-dir", inputs[0], 
                        "--output-dir", outputs[0]
                    ],
    )

def _evaluate_batch(artifacts_path, parquet_path):
    pandas_processor = get_pandas_processor("evaluate_batch")
    artifacts = ("/opt/ml/processing/artifacts", artifacts_path)
    inputs = ("/opt/ml/processing/input", parquet_path)
    outputs = ("/opt/ml/processing/output", f"s3://{S3_BUCKET}/model-evaluation")
   
    return ProcessingStep(
        name="EvaluateModel",
        processor=pandas_processor,
        inputs=[
            ProcessingInput(source=artifacts[1], destination=artifacts[0]),
            ProcessingInput(source=inputs[1], destination=inputs[0]),
        ],
        outputs=[
            ProcessingOutput(source=outputs[0], destination=outputs[1]),   
        ],
        code="scripts/evaluate_batch.py",
        job_arguments=["--input-dir", inputs[0], 
                        "--output-dir", outputs[0],
                        "--artifacts-dir", artifacts[0]
                    ],
    )


pipeline = Pipeline(
    name="FeatureEnggPipeline",
    steps=[
        # _tfdv_analyze(os.path.abspath("dataset/raw/3"), "false")
        # _fix_anomalies(os.path.abspath("dataset/raw/3")),
        #_convert_to_parquet(f"s3://{S3_BUCKET}/dataset/eval/csv", f"s3://{S3_BUCKET}/dataset/eval/parquet")
        #_train(f"s3://{S3_BUCKET}/dataset/parquet")
        _evaluate_batch(os.path.abspath("artifacts"), f"s3://{S3_BUCKET}/dataset/eval/parquet")
    ],
    sagemaker_session=local_session
)

pipeline.upsert(role_arn=ROLE)
execution = pipeline.start()
