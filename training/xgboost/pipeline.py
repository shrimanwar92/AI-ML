from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.local import LocalSession
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import boto3
import os

DEFAULT_REGION = "us-east-1"
ROLE = "DummyRole"  # pre-created IAM Role
S3_BUCKET = "sagemaker-tfdv-tft-demo"

os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION

boto_session = boto3.Session(region_name=DEFAULT_REGION)
local_session = LocalSession(boto_session=boto_session)
local_session.config = {"local": {"local_code": True}}

xgb_processor = ScriptProcessor(
    image_uri="pandas-image:latest",  # or use prebuilt XGBoost image
    role=ROLE,
    command=["python3"],
    instance_count=1,
    instance_type="local"
)

input_dir = "/opt/ml/input/data/train"
model_dir = "/opt/ml/model"

processing_step = ProcessingStep(
    name="TrainXGBoostModel",
    processor=xgb_processor,
    inputs=[
        ProcessingInput(source=f"s3://{S3_BUCKET}/dataset/transformed/", destination=input_dir)
    ],
    outputs=[
        ProcessingOutput(source=model_dir, destination=f"s3://{S3_BUCKET}/output/model/")
    ],
    code="steps/train_xgb.py",
    job_arguments=["--input-dir", input_dir , "--model-dir", model_dir],
)

pipeline = Pipeline(
    name="XGBoostPipeline",
    steps=[processing_step],
    sagemaker_session=local_session
)

pipeline.upsert(role_arn=ROLE)
execution = pipeline.start()