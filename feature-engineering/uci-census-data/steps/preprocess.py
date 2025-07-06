import pandas as pd
from zenml import step
from utils import RAW_DIR, CLEAN_DIR
import filecmp
from typing import List

@step(enable_cache=False)
def preprocess_data(raw_new_files) -> List[str]:
    cleaned_paths = []

    if len(raw_new_files) > 0:
        for file in raw_new_files:
            df = pd.read_csv(f"{RAW_DIR}/{file}", index_col=False)
            df.rename(columns={
                'education.num': 'education_num',
                'marital.status': 'marital_status',
                'capital.gain': 'capital_gain',
                'capital.loss': 'capital_loss',
                'hours.per.week': 'hours_per_week',
                'native.country': 'native_country'
            }, inplace=True)

            string_cols = df.select_dtypes(include='object').columns
            df[string_cols] = df[string_cols].fillna('unknown')

            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(0)
            
            cleaned_file = f"{CLEAN_DIR}/{file}"
            df.to_csv(cleaned_file, index=False)
            cleaned_paths.append(cleaned_file)
            print("Preprocessing complete.")
    
    return cleaned_paths