import os
from pathlib import Path
import filecmp
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATASET_DIR / 'raw'
CLEAN_DIR = DATASET_DIR / 'clean'
OUTPUT_DIR = PROJECT_ROOT / "outputs"
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to get feature by name
def get_feature_by_name(schema, name):
    for feature in schema.feature:
        if feature.name == name:
            return feature
    return None

def combine_csv(all_files):
    import pandas as pd
    
    list_of_dfs = []
    
    for filename in all_files:
        single_df = pd.read_csv(filename)
        list_of_dfs.append(single_df)

    return pd.concat(list_of_dfs, ignore_index=True)

def get_tfdv_schema() -> Tuple[str, str]:
    # Load from predefined paths or from logged artifacts
    return f"{OUTPUT_DIR}/schema.pbtxt", f"{OUTPUT_DIR}/baseline_stats.txt"

def get_new_csv():
    dcmp = filecmp.dircmp(RAW_DIR, CLEAN_DIR)
    has_matching_files = len(dcmp.left_only) == len(dcmp.right_only)
    print(list(dcmp.left_only))
    print("Do files match between raw and clean?", has_matching_files)
    if has_matching_files:
        raise ValueError("No new files to process. Stopping pipeline.")
    
    return list(dcmp.left_only)