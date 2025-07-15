import tensorflow as tf
print('TF: {}'.format(tf.__version__))
import tensorflow_data_validation as tfdv
print('TFDV version:', tfdv.version.__version__)
import argparse
from pathlib import Path
import glob

def _save_anomalies_to_md(output_path, anomalies, stem):
    anomalies_md_path = output_path / f"[{stem}]_anomalies_summary.md"
    print(f"Saving anomalies Readme.md to {anomalies_md_path}")
    with open(anomalies_md_path, 'w') as f:
        f.write(f"# Data Anomalies Detected in file {stem}\n\n")
        for feature, details in anomalies.anomaly_info.items():
            print(feature)
            f.write(f"### Feature: `{feature}`\n")
            f.write(f"- Description: {details.description}\n")
            f.write(f"- Severity: {details.severity}\n\n")
            for reason in details.reason:
                f.write(f"- Reason: {reason}\n\n")
                print(reason)

def _save_anomalies_pbtxt(output_path, anomalies, stem):
    anomalies_path = output_path / f'[{stem}]_anomalies.pbtxt'
    print(f"Saving anomalies pbtxt to {anomalies_path}")
    tfdv.write_anomalies_text(anomalies, anomalies_path)


def validate_csv(schema_dir, data_dir, output_dir) -> bool:
    print("Validating csv..")
    schema_path = Path(schema_dir)
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cleaned_csv = glob.glob(f"{data_path}/*.csv")
    schema = tfdv.load_schema_text(schema_path / "schema.pbtxt")
    baseline_stats = tfdv.load_statistics(schema_path / "baseline_stats.txt")

    for file in cleaned_csv:
        new_stats = tfdv.generate_statistics_from_csv(data_location=file)
        anomalies = tfdv.validate_statistics(
            statistics=new_stats,
            schema=schema,
            previous_statistics=baseline_stats
        )
        anomalies = tfdv.validate_statistics(
            statistics=new_stats,
            schema=schema,
            previous_statistics=baseline_stats
        )
        stem = file.split("/")[-1]
        _save_anomalies_pbtxt(output_path, anomalies, stem)
        _save_anomalies_to_md(output_path, anomalies, stem)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    validate_csv(args.schema_dir, args.data_dir, args.output_dir)
