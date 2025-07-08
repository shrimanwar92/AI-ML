from pipeline import tfdv_pipeline, fix_anomaly, transform
from utils import CLEAN_DIR


if __name__ == "__main__":
    #tfdv_pipeline("new")
    # files = [
    #     f"{CLEAN_DIR}/SRmopfnncL.csv"
    # ]
    # fix_anomaly(files)
    transform(csv_file=f"{CLEAN_DIR}/adult_drifted_invalid.csv", analyze=False)

