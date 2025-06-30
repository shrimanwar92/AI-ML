from preprocess import preprocess_data
from compute_tfdv_schema import compute_tfdv_schema
from validate_tfdv import validate_data
import glob
import filecmp
import os
from zenml import step, pipeline

SCHEMA_UPDATE = False
RAW_DIR = "../dataset/raw"
RAW_FILES = glob.glob(f"{RAW_DIR}/*.csv")
CLEAN_DIR = '../dataset/clean'
OUTPUT_DIR = "../outputs"
os.makedirs(CLEAN_DIR, exist_ok=True)

@step
def preprocess_data_step():
    dcmp = filecmp.dircmp(RAW_DIR, CLEAN_DIR)
    has_matching_files = len(dcmp.left_only) == len(dcmp.right_only)

    print("Do files match between raw and clean?", has_matching_files)
    
    if not has_matching_files:
        print("Preprocessing data...")
        preprocess_data(RAW_FILES)  # this is your imported function
        print("Finished data preprocessing.")

@step
def compute_stats_step():
    if SCHEMA_UPDATE == True:
        print("Computing baseline TFDV stats...")
        compute_tfdv_schema()
        print("Schema generated successfully.")

@step
def validate_new_data():
    validate_data()


@pipeline(enable_cache=False)
def full_pipeline():
    cleaned = preprocess_data_step()
    schema = compute_stats_step()
    #validate_new_data()  # you can pass inputs here if needed

if __name__ == "__main__":
    full_pipeline()

