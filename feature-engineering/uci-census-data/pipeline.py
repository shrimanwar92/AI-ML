from zenml import pipeline, step
from steps.preprocess import preprocess_data
from steps.compute_tfdv_schema import compute_tfdv_schema
from steps.tfdv_validate import validate_csv
from steps.fix_anomaly import fix_anomalies
from steps.transform import transform_data
from utils import RAW_DIR, CLEAN_DIR, get_new_csv, get_tfdv_schema
from typing import List
import glob    

@pipeline
def baseline_clean():
    raw_csvs = glob.glob(f"{RAW_DIR}/*.csv")
    cleaned_csvs = preprocess_data(raw_csvs)
    print("Baseline cleaning done.")
    print(cleaned_csvs)

@pipeline
def compute_schema():
    cleaned_csvs = glob.glob(f"{CLEAN_DIR}/*.csv")
    schema_path, stats_path = compute_tfdv_schema(cleaned_csvs)
    validate_csv(cleaned_csvs, schema_path, stats_path)

@pipeline
def validate_new_csv():
    new_csvs = get_new_csv()
    cleaned_csvs = preprocess_data(new_csvs)
    schema_path, stats_path = get_tfdv_schema()
    validate_csv(cleaned_csvs, schema_path, stats_path)

@pipeline
def fix_anomaly(csv_files: List[str]):
    fix_anomalies(csv_files)

@pipeline
def transform(csv_files: List[str], analyze: bool):
    transform_data(csv_files, analyze)


if __name__ == "__main__":
    #baseline_clean()
    #compute_schema()
    #validate_new_csv()
    #fix_anomaly([
        # f"{CLEAN_DIR}/SRmopfnncL.csv", 
        # f"{CLEAN_DIR}/adult.csv", 
        # f"{CLEAN_DIR}/adult_drifted_invalid.csv"
    #])
    transform(csv_files=[
        f"{CLEAN_DIR}/SRmopfnncL.csv", 
        f"{CLEAN_DIR}/adult.csv", 
        f"{CLEAN_DIR}/adult_drifted_invalid.csv"
    ], analyze=True)