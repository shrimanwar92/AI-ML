\# 📊 SageMaker Feature Engineering \& Validation Pipeline



This repository implements an \*\*end-to-end feature engineering, validation, and feature storage pipeline\*\* using \*\*Amazon SageMaker Processing jobs\*\*.  

It uses \*\*Pandas\*\* for preprocessing, \*\*TensorFlow Data Validation (TFDV)\*\* for schema generation and anomaly detection, and stores final features in \*\*SageMaker Feature Store\*\*.



---



\## 📌 Pipeline Steps



\### 1️⃣ Preprocess (`\_preprocess`)

\- Cleans raw CSV data.

\- Saves cleaned dataset to: `s3://<S3\_BUCKET>/dataset/cleaned`





\### 2️⃣ Compute Schema (`\_compute\_schema`)

\- Uses \*\*TFDV\*\* to:

\- Generate a schema from cleaned data.

\- Compute data statistics.

\- Saves schema to: `s3://<S3\_BUCKET>/schema`





\### 3️⃣ Validate CSV (`\_validate\_csv`)

\- Validates cleaned CSV data against the \*\*generated schema\*\*.

\- Outputs anomalies to: `s3://<S3\_BUCKET>/anomalies





\### 4️⃣ Fix Anomalies (`\_fix\_anomalies`)

\- Reads \*\*schema\*\*, \*\*cleaned data\*\*, and \*\*anomaly reports\*\*.

\- Corrects invalid values and fixes detected anomalies.

\- Updates:

\- `s3://<S3\_BUCKET>/dataset/cleaned`

\- `s3://<S3\_BUCKET>/anomalies`



\### 5️⃣ Transform (`\_transform`)

Two modes:

\- \*\*Analyze Mode (`analyze=True`)\*\*

\- Runs TFDV analysis \& transformation.

\- Saves transformed dataset and transformation function.

\- \*\*Inference Mode (`analyze=False`)\*\*

\- Applies a previously saved transformation function to new data.



\### 6️⃣ Store Features in Feature Store (`\_store\_features\_in\_feature\_store`)

\- Reads transformed dataset.

\- Stores features in \*\*Amazon SageMaker Feature Store\*\*.



---



\## 🔄 Pipeline Flow



\### Overall Pipeline

```mermaid

flowchart TD

&nbsp; A\[Raw CSV in S3] --> B\[Preprocess]

&nbsp; B --> C\[Cleaned Dataset in S3]

&nbsp; C --> D\[Compute Schema]

&nbsp; D --> S1\[Schema in S3]

&nbsp; C --> E\[Validate CSV]

&nbsp; S1 --> E

&nbsp; E --> F\[Anomalies in S3]

&nbsp; F --> G\[Fix Anomalies]

&nbsp; G --> H\[Updated Cleaned Dataset in S3]

&nbsp; H --> I\[Transform Data]

&nbsp; S2\[Transform Fn in S3] -.-> I

&nbsp; I --> J\[Transformed Dataset in S3]

&nbsp; J --> K\[Store Features in Feature Store]



\### Transform Flow — Analyze Mode (analyze=True)

```mermaid

flowchart TD

&nbsp;   A\[Cleaned Dataset in S3] --> B\[Analyze + Transform]

&nbsp;   B --> C\[Transformed Dataset in S3]

&nbsp;   B --> D\[Save Transform Function in S3]





\### Transform Flow — Inference Mode (analyze=False)

```mermaid

flowchart TD

&nbsp;   A\[Cleaned Dataset in S3] --> B\[Apply Existing Transform Function]

&nbsp;   T\[Transform Function in S3] --> B

&nbsp;   B --> C\[Transformed Dataset in S3]



\### 📂 Project Structure

```

.

├── dataset/                  # Local dataset folder (raw/cleaned)

├── steps/

│   ├── preprocess.py         # Data preprocessing

│   ├── compute\_tfdv\_schema.py# Generate schema \& stats

│   ├── tfdv\_validate.py      # Validate data

│   ├── fix\_anomaly.py        # Fix anomalies

│   ├── transform.py          # Transform features

│   ├── store\_features\_in\_feature\_store.py # Store features in Feature Store

├── helpers.py                # Utility functions for SageMaker processors

├── pipeline.py               # Main pipeline code

└── README.md





