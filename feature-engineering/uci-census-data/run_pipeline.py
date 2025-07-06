from pipeline import tfdv_pipeline, fix_anomaly
from utils import CLEAN_DIR


if __name__ == "__main__":
    #tfdv_pipeline("new")
    files = [
        f"{CLEAN_DIR}/SRmopfnncL.csv"
    ]
    fix_anomaly(files)

