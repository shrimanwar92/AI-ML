import modin.config as modin_cfg
modin_cfg.Engine.put("ray")

import modin.pandas as mpd
import os
import glob
import re
import xgboost as xgb
import argparse

def load_feature_label_pairs(folder: str):
    feature_files = glob.glob(os.path.join(folder, "*__features.parquet"))
    label_files = glob.glob(os.path.join(folder, "*__labels.parquet"))

    feature_map = {re.search(r"transformed_(.*?)__features.parquet", f).group(1): f for f in feature_files}
    label_map = {re.search(r"transformed_(.*?)__labels.parquet", f).group(1): f for f in label_files}

    X_list, y_list = [], []
    for id_key in feature_map.keys() & label_map.keys():
        print(f"Loading pair for ID: {id_key}")
        X = mpd.read_parquet(feature_map[id_key])
        y = mpd.read_parquet(label_map[id_key])

        X = X.drop(columns=[col for col in ["record_id", "event_time"] if col in X.columns])
        y = y.drop(columns=[col for col in ["record_id", "event_time"] if col in y.columns])

        X_list.append(X)
        y_list.append(y)

    X_all = mpd.concat(X_list)
    y_all = mpd.concat(y_list)
    return X_all.to_numpy(), y_all.to_numpy().ravel()

def train(input_dir, model_dir):
    X, y = load_feature_label_pairs(input_dir)

    # Train
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=100)
    # Save model
    model.save_model(f"{model_dir}/xgboost-model.bst")
    print(f"✅ Saved model to: {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--model-dir", type=str)
    args = parser.parse_args()
    print(args)

    train(args.input_dir, args.model_dir)