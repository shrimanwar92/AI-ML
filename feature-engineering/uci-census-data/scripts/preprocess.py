import pandas as pd
import os
from pathlib import Path

def preprocess_data(raw_files):
    from run_pipeline import CLEAN_DIR

    for file in raw_files:
        df = pd.read_csv(file, index_col=False)

        string_cols = df.select_dtypes(include='object').columns
        df[string_cols] = df[string_cols].fillna('unknown')

        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)
    
        p = Path(file)
        df.to_csv(f"{CLEAN_DIR}/{p.stem}.csv", index=False)