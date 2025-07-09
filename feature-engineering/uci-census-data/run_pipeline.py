from pipeline import baseline_clean, compute_schema, validate_new_csv, fix_anomaly, transform
from utils import CLEAN_DIR


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