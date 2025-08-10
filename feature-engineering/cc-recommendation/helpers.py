from sagemaker.local import LocalSession
import boto3
from sagemaker.processing import ScriptProcessor

DEFAULT_REGION = "us-east-1"
ROLE = "DummyRole" # this is actually created in AWS
S3_BUCKET = "cc-recommendation"

# ✅ LocalSession uses mocked boto3 session
boto_session = boto3.Session(region_name=DEFAULT_REGION)
local_session = LocalSession(boto_session=boto_session)
local_session.config = {"local": {"local_code": True}}
    

def get_pandas_processor(base_job_name = "preprocess"):
    return ScriptProcessor(
        image_uri="pandas-image:latest",
        command=["python3"],
        instance_type="local",
        instance_count=1,
        base_job_name=base_job_name,
        role=ROLE,
        sagemaker_session=local_session,
    )

def get_tensorflow_processor(job_name = "preprocess"):
    # ✅ ScriptProcessor with your custom Docker image
    return ScriptProcessor(
        image_uri="tfdv-tft:latest",
        command=["python3"],
        instance_type="local",
        instance_count=1,
        base_job_name=job_name,
        role=ROLE,
        sagemaker_session=local_session,
    )