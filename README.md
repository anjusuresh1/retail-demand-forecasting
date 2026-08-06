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
