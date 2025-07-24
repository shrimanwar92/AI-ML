import uuid
import time
import pyarrow.parquet as pq
import pyarrow.fs
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.local import LocalSession
import argparse
import re
import boto3
from sagemaker.session import Session

# --- CONFIG ---
region = "us-east-1"
feature_group_name = "uci-census-offline-only-fg"
record_identifier_name = "record_id"
event_time_feature_name = "event_time"

# --- SETUP ---
fs = pyarrow.fs.S3FileSystem(region=region)
# boto_session = boto3.Session(region_name=region)
# session = LocalSession(boto_session=boto_session)
# session.config = {"local": {"local_code": True}}
boto_session = boto3.Session(region_name=region)
session = Session(boto_session=boto_session)

def _create_feature_group(feature_group, s3_parquet_files, role_arn):
    first_file = pq.read_table(s3_parquet_files[0], filesystem=fs)
    schema = first_file.schema
    feature_definitions = [
        {"FeatureName": col.name, "FeatureType": "Fractional" if str(col.type) in ["double", "float"] else "String"}
        for col in schema
    ]
    feature_definitions += [
        {"FeatureName": record_identifier_name, "FeatureType": "String"},
        {"FeatureName": event_time_feature_name, "FeatureType": "String"},
    ]
    feature_group.create(
        feature_definitions=feature_definitions,
        record_identifier_name=record_identifier_name,
        event_time_feature_name=event_time_feature_name,
        role_arn=role_arn,
        online_store_config={"EnableOnlineStore": False},
        offline_store_config={
            "S3StorageConfig": {
                "S3Uri": f"s3://{session.default_bucket()}/feature-store/{feature_group_name}"
            }
        }
    )
    feature_group.wait_for_create()
    print("Feature group created.")


def store_features(role_arn, files, data_dir):
    # --- Create FG if needed ---
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)
    file_paths = [f"{data_dir}/{file}" for file in files]
    
    try:
        feature_group.describe()
    except Exception as e:
        # Infer schema from first parquet file
        _create_feature_group(feature_group, file_paths, role_arn)

    for file_uri in file_paths:
        print(f"📦 Loading: {file_uri}")

        # Parse metadata
        match = re.match(r"transformed_(.*?)__(features|labels)\.parquet", file_uri)
        if not match:
            continue
        source_id, file_type = match.groups()
        
        table = pq.read_table(file_uri, filesystem=fs)
        df = table.to_pandas()

        # Add metadata
        df["file_id"] = source_id
        df["file_uri"] = file_uri

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        df[record_identifier_name] = [str(uuid.uuid4()) for _ in range(len(df))]
        df[event_time_feature_name] = now

        print(f"⬆️ Ingesting {len(df)} rows to offline store...")
        feature_group.ingest(data_frame=df, max_workers=8, wait=True)
        print("✅ Ingested.")

    print("🎉 All files processed and pushed to SageMaker Offline Feature Store.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--files", nargs='+', help="List of files to process")
    args = parser.parse_args()

    store_features(args.role_arn, args.files, args.data_dir)