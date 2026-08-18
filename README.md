# Retail Demand Forecasting & Pricing Optimization

## Project Overview

This project builds an end-to-end machine learning system for retail demand forecasting and price optimization. The system predicts future product demand and recommends optimal pricing strategies to maximize revenue and profit margins.

## Business Impact

- **Accurate demand forecasting** reduces inventory costs and stockouts
- **Dynamic pricing optimization** increases profit margins by 10-20%
- **Data-driven decisions** replace manual, error-prone processes

## Dataset

- **Source:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/592/online+retail+ii)
- **Time period:** 2009-2011
- **Records:** ~1M transactions
- **Features:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, Price, CustomerID, Country

## Project Structure
retail-demand-forecasting/
├── data/
│ ├── raw/ # Raw, immutable data
│ └── processed/ # Cleaned, transformed data
├── src/
│ ├── data/ # ETL pipelines
│ ├── features/ # Feature engineering
│ ├── models/ # Model training & inference
│ └── api/ # FastAPI application
├── notebooks/ # Exploratory analysis
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
└── README.md


## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data

Download the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/592/online+retail+ii) and save to `data/raw/online_retail_II.xlsx`.

### 3. Run ETL Pipeline

```bash
cd src/data
python etl.py
```

### 4. Verify Processed Data

```python
import pandas as pd
df = pd.read_parquet('../data/processed/sales_daily.parquet')
print(df.head())

## Day 2 Baseline Results

The validation set contains the final 28 days of the dataset. 
The split is chronological to prevent future information from entering training.

          model       MAE       RMSE       MAPE       WAPE
    NaiveWeekly 28.528971 128.780580 337.720462 116.230201
RidgeRegression 21.494573  61.470774 356.981352  89.949986
        XGBoost 22.096281  83.882901 290.281295  92.467998

The selected model for the next stage will be based on validation performance,
business interpretability, inference speed, and operational complexity.

## Day 3: Advanced Modeling

### Models Evaluated

The following XGBoost configurations were evaluated:

- Baseline XGBoost
- Regularized XGBoost
- Shallow XGBoost

The validation set contained the final 28 days of observations. A chronological split was used to avoid future-data leakage.

### Experiment Tracking

MLflow recorded:

- Hyperparameters
- MAE
- RMSE
- MAPE
- WAPE
- Trained model artifacts

Run the MLflow UI with:

```bash
python -m mlflow ui
```

### Model Artifacts

The selected model is saved to:

```text
models/tuned_xgboost.joblib
```

Feature importance is saved to:

```text
models/feature_importance.csv
models/feature_importance.png
```

### Model Selection

The final model was selected using validation WAPE, while also considering inference speed, interpretability, and deployment complexity.

Actual validation metrics are reported below:

         model       MAE      RMSE       MAPE       WAPE
xg_regularized 18.102295 73.699718 310.231954  88.062210
   xg_baseline 18.549111 70.608182 319.412183  90.235837
    xg_shallow 21.245170 77.234983 408.358883 103.351348