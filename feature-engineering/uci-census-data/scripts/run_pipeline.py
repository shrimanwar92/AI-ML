from preprocess import preprocess_data
from compute_tfdv_schema import compute_tfdv_schema
from validate_tfdv import validate_data
import glob
import filecmp
import os

SCHEMA_UPDATE = True
RAW_DIR = "../dataset/raw"
RAW_FILES = glob.glob(f"{RAW_DIR}/*.csv")
CLEAN_DIR = '../dataset/clean'
OUTPUT_DIR = "../outputs"
os.makedirs(CLEAN_DIR, exist_ok=True)

def main():
    dcmp = filecmp.dircmp(RAW_DIR, CLEAN_DIR)
    has_matching_files = len(dcmp.left_only) == len(dcmp.right_only)

    print(has_matching_files)
    
    if has_matching_files == False:
        print("Preprocessing data...")
        preprocess_data(RAW_FILES)
        print("Finished data preprocessing.")
    
    if SCHEMA_UPDATE == True:
        print("Computing baseline TFDV stats...")
        compute_tfdv_schema()
        print("Schema generated successfully.")
    
    #validate_data()

if __name__ == "__main__":
    main()
