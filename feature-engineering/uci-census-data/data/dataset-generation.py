import pandas as pd
import numpy as np
import random
import string

# Load original dataset
df = pd.read_csv("./raw/adult.csv")

# Target total rows
TARGET_ROWS = 1000

# Calculate how many times to replicate
repeats = TARGET_ROWS // len(df) + 1
df_big = pd.concat([df] * repeats, ignore_index=True)
df_big = df_big.head(TARGET_ROWS)

# --- Inject random missing values ---
def insert_missing_values(df, prob=0.01):
    for col in df.columns:
        mask = np.random.rand(len(df)) < prob
        df.loc[mask, col] = np.nan
    return df

# --- Inject invalid string values ---
def insert_invalid_values(df, col, invalid_vals, prob=0.01):
    mask = np.random.rand(len(df)) < prob
    df.loc[mask, col] = np.random.choice(invalid_vals, size=mask.sum())
    return df

# --- Drift numeric features ---
def drift_numeric_feature(df, col, drift_func):
    num_rows = len(df)
    drift_rows = int(0.3 * num_rows)  # drift 30%
    drift_indices = np.random.choice(df.index, size=drift_rows, replace=False)
    df.loc[drift_indices, col] = drift_func(df.loc[drift_indices, col])
    return df

# Introduce missing values
df_big = insert_missing_values(df_big, prob=0.02)

# Inject invalid entries
df_big = insert_invalid_values(df_big, "education", ["??", "unknown", 123], prob=0.03)
df_big = insert_invalid_values(df_big, "workclass", ["n/a", "###", None], prob=0.02)
df_big = insert_invalid_values(df_big, "occupation", ["bad_data", "!", "missing"], prob=0.03)

# Drift numeric features
df_big = drift_numeric_feature(df_big, "age", lambda x: x + np.random.randint(10, 30, size=len(x)))
df_big = drift_numeric_feature(df_big, "hours.per.week", lambda x: x * np.random.uniform(0.5, 1.5, size=len(x)))

# Shuffle rows
df_big = df_big.sample(frac=1).reset_index(drop=True)

random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

# Save to new CSV
df_big.to_csv(f"./raw/{random_string}.csv", index=False)

print("✅ Generated 'adult_drifted_invalid.csv' with 100,000 rows including drifted and invalid data.")
