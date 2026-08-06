"""
    ETL Pipeline for Retail Demand Forecasting
    
    This module will handle
    - Data Extraction from raw files
    - Data cleaning and validation
    - Feature Creation and transformation
    - Saving processed data for model training and evaluation
    
    Author: Anju Suresh
    Date: 4-8-2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

#configure logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def load_raw_data(file_path:str) -> pd.DataFrame:
    """
    Load raw data from the specified file path.
    
    Args:
        file_path (str): Path to the raw data file.
    
    Returns:
        return dataframe with raw data 
    """
    logger.info(f"Loading raw data from {file_path}")
    df = pd.read_excel(file_path)
    logger.info(f'loaded data with rows: {len(df)} and columns: {len(df.columns)}')
    return df

def data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the data for missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
    
    Returns:
        pd.DataFrame: Validated dataframe.
    """
    logger.info('Validating the data')
    initialRows = len(df)
    
    #Normalize StockCode to string to avoid mixed-type parquet issues
    df['StockCode'] = df['StockCode'].astype(str)
    
    #Drop rows with missing Customer ID (needed for grouping)
    df = df.dropna(subset=['Customer ID'])
    df['Customer ID'] = df['Customer ID'].astype(int)
    logger.info(f'Dropped {initialRows - len(df)} rows with missing Customer ID')
    
#Remove rows with missing values in critical columns
    df = df.dropna(subset=['Invoice', 'StockCode', 'Quantity', 'InvoiceDate', 'Price'])
    logger.info(f'Remaining rows after dropping missing critical values: {len(df)}')
    
    #Remove duplicate rows
    df = df.drop_duplicates()
    logger.info(f'Removed duplicates, remaining rows: {len(df)}')
    
    #Filter data from products with price and quantity below or equal to zero
    df = df[(df['Price'] > 0) & (df['Quantity'] > 0)]
    logger.info(f'Filtered products with price <= 0 or quantity <= 0, remaining rows: {len(df)}')
    
    #Filter data from cancelled orders (Invoice starting with 'C')
    df['Invoice'] = df['Invoice'].astype(str)  # Cast to string before string op
    df = df[~df['Invoice'].str.startswith('C')]
    logger.info(f'Filtered cancelled orders, remaining rows: {len(df)}')
    
    #Convert date column to datetime format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    #Create Revenue column
    df['Revenue'] = df['Quantity'] * df['Price']
    logger.info('Data validation completed')
    
    return df

def create_customer_product_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data on Customer, product, date for demand forecasting

    Args:
        df (pd.DataFrame)

    Returns:
        Aggregated DataFrame
    """
    df['Date'] = df['InvoiceDate'].dt.date
    aggregated = df.groupby(['Customer ID', 'StockCode', 'Date']).agg({
        'Quantity': 'sum',
        'Price': 'mean',
        'Revenue': 'sum',
        'Description': 'first',
        'Country': 'first'
    }).reset_index()
    logger.info(f'Created customer-product level data with rows: {len(aggregated)}')
    return aggregated

def time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time based features for forecasting

    Args:
        df (pd.DataFrame): _description_

    Returns:
        Dataframe with time based features
    """
    logger.info('Adding time based features')
    
    #Converting date to datetime for feature engineering
    df['Date'] = pd.to_datetime(df['Date'])
    
    #Extract time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    #Flag for Holiday
    df['IsHoliday'] = 0
    
    logger.info('Time based features added')
    return df
    
def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save Processed Data to parquet format
    
    Args:
        df (pd.Dataframe): data to save
        output_path (str): path to save the processed data
    """
    #Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    #Save to parquet
    df.to_parquet(output_path, index=False)
    
    logger.info(f'Processed data with {len(df)} rows saved to {output_path}')
    
def run_etl_pipeline(raw_data_path: str, processed_data_path: str) -> None:
    """
    Run the complete ETL Pipelines
    Args:
        raw_data_path (str): _description_
        processed_data_path (str): _description_
    """
    logger.info('=' * 50)
    logger.info('Starting ETL Pipeline')
    logger.info('=' * 50)

    #load raw data
    df_raw = load_raw_data(raw_data_path)
    
    #Data validation
    df_clean = data_validation(df_raw)
    
    #Aggragte data to customer-product level
    df_aggregated = create_customer_product_level(df_clean)
    
    #Add time-based features
    df_final = time_based_features(df_aggregated)
    
    #Save processed data
    save_processed_data(df_final, processed_data_path)
    
    logger.info('=' * 50)
    logger.info('ETL Pipeline Completed Successfully')
    logger.info('=' * 50)

    return df_final

if __name__ == "__main__":
    # Build paths relative to this script so it works regardless of CWD
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../retail-demand-forecasting
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
    PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sales_daily.parquet"
    
    #Run ETL Pipeline
    df_processed = run_etl_pipeline(str(RAW_DATA_PATH), str(PROCESSED_PATH))
    
    #Quick Verification
    print("\nProcessed Data Summary")
    print(f"Shape: {df_processed.shape}")
    print(f"Data range from {df_processed['Date'].min()} to {df_processed['Date'].max()}")
    print(f"Unique Customers: {df_processed['Customer ID'].nunique()}")
    print(f"Unique Products: {df_processed['StockCode'].nunique()}")
    print(f"\nFirst 5 Rows: \n{df_processed.head(5)}")
