\# 🛠 ZenML Data Processing Pipelines



This repository implements a \*\*data validation and transformation workflow\*\* using \*\*ZenML pipelines\*\* with \*\*TensorFlow Data Validation (TFDV)\*\* for schema generation and anomaly detection.  



The pipelines handle:

\- \*\*Data Cleaning\*\*

\- \*\*Schema Computation\*\*

\- \*\*Validation of New Data\*\*

\- \*\*Anomaly Fixing\*\*

\- \*\*Feature Transformation\*\*



---



\## 📌 Pipelines Overview



\### 1️⃣ Baseline Clean (`baseline\_clean`)

\- Reads \*\*raw CSV files\*\* from `RAW\_DIR`.

\- Preprocesses and cleans data.

\- Outputs cleaned CSVs into `CLEAN\_DIR`.



---



\### 2️⃣ Compute Schema (`compute\_schema`)

\- Reads cleaned CSVs from `CLEAN\_DIR`.

\- Generates \*\*TFDV schema\*\* and \*\*statistics\*\*.

\- Validates data against the generated schema.



---



\### 3️⃣ Validate New CSV (`validate\_new\_csv`)

\- Fetches \*\*new incoming CSVs\*\*.

\- Cleans them using preprocessing step.

\- Loads \*\*existing schema\*\* and \*\*stats\*\*.

\- Validates new data for anomalies.



---



\### 4️⃣ Fix Anomaly (`fix\_anomaly`)

\- Reads CSV files with anomalies.

\- Applies fixes (e.g., correcting categories, handling missing values).

\- Saves cleaned versions.



---



\### 5️⃣ Transform (`transform`)

\- Applies \*\*feature engineering and transformations\*\*.

\- Can optionally run \*\*analysis mode\*\* before transformation.



---



\## 🔄 Workflow Diagram



```mermaid

flowchart TD

&nbsp;   A\[Raw CSVs in RAW\_DIR] --> B\[Baseline Clean]

&nbsp;   B --> C\[Cleaned CSVs in CLEAN\_DIR]

&nbsp;   C --> D\[Compute Schema]

&nbsp;   D --> S1\[TFDV Schema + Stats]

&nbsp;   D -->|Validate| E\[Validated Clean Data]

&nbsp;   

&nbsp;   subgraph New Data Validation

&nbsp;       N1\[New CSVs] --> N2\[Preprocess]

&nbsp;       N2 --> N3\[Load Existing Schema + Stats]

&nbsp;       N3 --> N4\[Validate New Data]

&nbsp;   end



&nbsp;   subgraph Anomaly Handling

&nbsp;       A1\[CSV with Anomalies] --> A2\[Fix Anomaly Step]

&nbsp;       A2 --> A3\[Cleaned Fixed CSV]

&nbsp;   end



&nbsp;   subgraph Feature Transformation

&nbsp;       F1\[Cleaned CSVs] --> F2\[Transform Step]

&nbsp;       F2 --> F3\[Transformed Features]

&nbsp;   end



\## 📂 Project Structure

.

├── steps/

│   ├── preprocess.py          # Data cleaning and preprocessing

│   ├── compute\_tfdv\_schema.py # Schema + stats computation

│   ├── tfdv\_validate.py       # TFDV validation

│   ├── fix\_anomaly.py         # Fixes detected anomalies

│   ├── transform.py           # Feature transformation logic

├── utils.py                   # Helper functions and constants

├── pipelines.py               # ZenML pipeline definitions

├── README.md                  # Documentation





🚀 Running Pipelines

Baseline Cleaning

zenml pipeline run baseline\_clean



Compute Schema

zenml pipeline run compute\_schema



Validate New Data

zenml pipeline run validate\_new\_csv



Fix Anomalies

zenml pipeline run fix\_anomaly --csv\_files data/anomalies/\*.csv



Transform Data

zenml pipeline run transform --csv\_files data/clean/\*.csv --analyze true



