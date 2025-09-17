# 🚖 NYC Taxi ETA Prediction – End-to-End AWS ML Pipeline (POC)

<img width="1883" height="222" alt="architecture" src="https://github.com/user-attachments/assets/816b2fba-d360-4e1e-8577-19b62863743b" />

## 📌 Project Overview
This project builds an **end-to-end machine learning pipeline** to predict **Estimated Time of Arrival (ETA)** for NYC Yellow Taxi trips using **AWS services**.

It demonstrates how to:
- Ingest raw parquet data from **Amazon S3**
- Automatically detect schema using **AWS Glue Crawlers** & store metadata in **AWS Glue Data Catalog**
- Query data interactively with **Amazon Athena**
- Preprocess, engineer features, and train an **XGBoost regression model** with **Amazon SageMaker**
- Evaluate model performance (MAE) and generate error samples
- Visualize results with Tableau / Superset

This POC follows **production-grade design principles** for scalability, automation, and reproducibility.

---

## 🏗 Architecture

**Key Components:**
1. **Amazon S3** – Raw data storage (`yellow_tripdata_*.parquet`) + derived CSV outputs  
2. **AWS Glue Crawler** – Infers schema and updates Glue Data Catalog  
3. **AWS Glue Data Catalog** – Central metadata repository for Athena & downstream use  
4. **Amazon Athena** – SQL queries for validation, exploration  
5. **Amazon SageMaker** – Managed XGBoost training job + model artifact storage  
6. **SageMaker Model Artifact** – Downloaded locally for validation & batch inference  
7. **Visualization Tool** – Tableau (or Superset) for dashboarding  

---

## 🔧 Tech Stack

| Layer             | Tools Used |
|------------------|-----------|
| **Data Storage** | Amazon S3 |
| **ETL / Metadata** | AWS Glue (Crawler + Data Catalog) |
| **Query** | Amazon Athena |
| **Model Training** | Amazon SageMaker (Built-in XGBoost) |
| **Visualization** | Tableau / Apache Superset |
| **Languages & Libraries** | Python, Pandas, PyArrow, scikit-learn, xgboost |

---

## 📂 Project Workflow

### 1️⃣ Data Ingestion
- Upload NYC Yellow Taxi data (`yellow_tripdata_YYYY-MM.parquet`) to **S3**.
- Run **AWS Glue Crawler** to infer schema and store in **Glue Data Catalog**.

### 2️⃣ Data Exploration
- Query with **Amazon Athena** to:
  - Validate schema
  - Inspect row counts, nulls
  - Perform initial feature profiling

### 3️⃣ Feature Engineering & Preprocessing
- Convert timestamps to `datetime`
- Compute **trip duration (minutes)**
- Filter outliers: duration < 1 min or > 2 hrs, negative fares
- Derive features:
  - `hour`
  - `weekday`
  - `is_airport`
  - `has_congestion_fee`

### 4️⃣ Train/Validation Split
- Create **80/20 split**, label-first CSV (no headers) for XGBoost.
- Upload to S3 under:
  - `derived/eta_poc/train/`
  - `derived/eta_poc/val/`

### 5️⃣ Model Training (Managed XGBoost)

```python
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

xgb_img = image_uris.retrieve("xgboost", region=region, version="1.7-1")

estimator = Estimator(
    image_uri=xgb_img,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    hyperparameters={
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "eta": 0.2,
        "subsample": 0.8,
        "num_round": 120,
    }
)

estimator.fit({
    "train": TrainingInput(train_s3_uri, content_type="text/csv"),
    "validation": TrainingInput(val_s3_uri, content_type="text/csv"),
})
```

### 6️⃣ Model Evaluation (Local)
- **Download** `model.tar.gz` from S3  
- **Extract & Load** with `xgboost.Booster`  
- **Compute** Mean Absolute Error (MAE) on validation set  
- **Generate** error sample CSV (`eta_errors_model_sample.csv`) for visualization  

### 7️⃣ Visualization
- Load `eta_errors_model_sample.csv` into **Tableau** or **Apache Superset**
- Build dashboards for:
  - **Actual vs Predicted ETA**
  - **Hourly Performance**
  - **Error Distribution by Distance**

---

## 📊 Key Metrics

- **Train/Validation Size:** `96k / 24k`
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Baseline MAE (median binning):** ~`X.XX min`
- **Model MAE:** ~`Y.YY min` (**Z% improvement** over baseline)

---

## 🚀 Deployment & Scaling

- Extendable to **real-time inference** with **SageMaker Endpoints**
- Scales to **multi-GB datasets** using **SageMaker Processing** & distributed training
- **AWS Glue Workflows** can be scheduled for **automated retraining pipelines**

---

## 📌 Resume-Worthy Highlights

- **Designed & deployed an end-to-end ML pipeline on AWS**  
  (S3 → Glue → Athena → SageMaker)
- **Trained a regression model with XGBoost** to predict NYC taxi ETAs with production-level performance
- **Built interactive dashboards** (Tableau / Superset) to monitor model performance

---

## 🛠 Future Improvements

- Integrate **real-time inference** via SageMaker Endpoint + Lambda + API Gateway
- Add **CI/CD pipeline** for automated retraining
- Deploy **Superset / QuickSight dashboards** as shareable web apps

---

## 📜 License

This project is provided as a **proof of concept** and can be freely modified for learning and internal use.
