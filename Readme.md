🚖 NYC Taxi ETA Prediction – End-to-End AWS ML Pipeline (POC)

📌 Project Overview

This project builds an end-to-end machine learning pipeline to predict Estimated Time of Arrival (ETA) for NYC Yellow Taxi trips using AWS services.

It demonstrates how to:

Ingest raw parquet data from Amazon S3

Automatically detect schema using AWS Glue Crawlers & store metadata in AWS Glue Data Catalog

Query data interactively with Amazon Athena

Preprocess, engineer features, and train an XGBoost regression model with Amazon SageMaker

Evaluate model performance (MAE) and generate error samples

Visualize results with Tableau / Superset

This POC follows production-grade design principles for data pipelines, scalability, and reproducibility.

🏗 Architecture

Key Components:

Amazon S3 – Raw data storage (yellow_tripdata_*.parquet) + derived CSV outputs

AWS Glue Crawler – Infers schema and updates Glue Data Catalog

AWS Glue Data Catalog – Central metadata repository for Athena & downstream use

Amazon Athena – SQL queries for validation, exploration

Amazon SageMaker – Managed XGBoost training job + model artifact storage

SageMaker Model Artifact – Downloaded locally for validation & batch inference

Visualization Tool – Tableau (or Superset) for dashboarding

🔧 Tech Stack
Layer	Tools Used
Data Storage	Amazon S3
ETL / Metadata	AWS Glue (Crawler + Data Catalog)
Query	Amazon Athena
Model Training	Amazon SageMaker (Built-in XGBoost)
Visualization	Tableau / Apache Superset
Languages & Libraries	Python, Pandas, PyArrow, scikit-learn, xgboost
📂 Project Workflow
1️⃣ Data Ingestion

NYC Yellow Taxi data (yellow_tripdata_YYYY-MM.parquet) uploaded to S3.

Glue crawler runs and updates schema in Glue Data Catalog.

2️⃣ Data Exploration

Athena queries used to explore data distribution, check nulls, and validate schema.

3️⃣ Feature Engineering & Preprocessing

Convert timestamps to datetime

Compute trip duration in minutes

Filter outliers (duration < 1 min or > 2 hrs, negative fares)

Derive features: hour, weekday, is_airport, has_congestion_fee

4️⃣ Train/Validation Split

80/20 split, label-first CSV format (no headers) for XGBoost.

Uploaded to S3 under derived/eta_poc/train/ and val/.

5️⃣ Model Training
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

6️⃣ Model Evaluation (Local)

Download model.tar.gz from S3, extract, and load with xgboost.Booster

Compute MAE on validation set

Generate error sample CSV for visualization

7️⃣ Visualization

Load eta_errors_model_sample.csv into Tableau / Superset

Build dashboards for:

Actual vs Predicted ETA

Hourly performance

Error distribution by distance

📊 Key Metrics

Train/Validation Size: 96k / 24k

Evaluation Metric: Mean Absolute Error (MAE) in minutes

Baseline MAE (median binning): ~X.XX min

Model MAE: ~Y.YY min (improvement of Z%)

🚀 Deployment & Scaling

Extendable to real-time inference with SageMaker Endpoints

Scales to multi-GB datasets using SageMaker Processing & distributed training

Glue workflows can be scheduled for automated retraining pipelines
