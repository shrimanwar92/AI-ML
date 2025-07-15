from sagemaker.local import LocalSession
import boto3

DEFAULT_REGION = "us-east-1"
ROLE = "DummyRole" # this is actually created in AWS
S3_BUCKET = "sagemaker-tfdv-tft-demo"

def setup_sagemaker_local_session():
    # ✅ LocalSession uses mocked boto3 session
    boto_session = boto3.Session(region_name=DEFAULT_REGION)
    local_session = LocalSession(boto_session=boto_session)
    local_session.config = {"local": {"local_code": True}}

    return local_session