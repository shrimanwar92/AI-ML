from zenml import pipeline, step
from steps.preprocess import preprocess_data
from steps.compute_tfdv_schema import compute_tfdv_schema
from steps.tfdv_validate import validate_csv
from steps.fix_anomaly import fix_anomalies
from utils import RAW_DIR, CLEAN_DIR, OUTPUT_DIR
import filecmp
from typing import Tuple, List

dcmp = filecmp.dircmp(RAW_DIR, CLEAN_DIR)

@step(enable_cache=False)
def skip_compute_schema(cleaned_csv: List[str]) -> Tuple[str, str]:
    # Load from predefined paths or from logged artifacts
    return f"{OUTPUT_DIR}/schema.pbtxt", f"{OUTPUT_DIR}/baseline_stats.txt"

@step(enable_cache=False)
def is_new_csv_present():
    has_matching_files = len(dcmp.left_only) == len(dcmp.right_only)
    print(list(dcmp.left_only))
    print("Do files match between raw and clean?", has_matching_files)
    if has_matching_files:
        raise ValueError("No new files to process. Stopping pipeline.")
    

@pipeline
def tfdv_pipeline(mode: str):
    if mode == "baseline":
        is_new_csv_present()
        cleaned_csvs = preprocess_data(list(dcmp.left_only))
        schema_path, stats_path = compute_tfdv_schema(cleaned_csvs)
        validate_csv(cleaned_csvs, schema_path, stats_path)
    elif mode == "new":
        is_new_csv_present()
        cleaned_csvs = preprocess_data(list(dcmp.left_only))
        schema_path, stats_path = skip_compute_schema(cleaned_csvs)
        validate_csv(cleaned_csvs, schema_path, stats_path)

@pipeline
def fix_anomaly():
    fix_anomalies()

