from pathlib import Path
import pandas as pd
import numpy as np

TARGET_COLUMN = 'demand'

FEATURE_COLUMNS = [
    'lag_1',
    'lag_7',
    'lag_14',
    'lag_28',
    'rolling_mean_7',
    'rolling_mean_14',
    'rolling_std_7',
    'rolling_max_7',
    'day_of_week',
    'day_of_month',
    'month',
    'week_of_year',
    'quarter',
    'is_weekend',
    'price',
    'average_price_7'
]

def create_calendar_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features from data"""
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['month'] = data['Date'].dt.month
    data['day_of_month'] = data['Date'].dt.day
    data['week_of_year'] = data['Date'].dt.isocalendar().week.astype(int)
    data['quarter'] = data['Date'].dt.quarter
    data['is_weekend'] = (data['day_of_week']>=5).astype(int)
    
    return data

def add_lag_features(df: pd.DataFrame, target_column: int = TARGET_COLUMN) -> pd.DataFrame:
    """Add historical demand features for each product"""
    data = df.copy()
    
    grouped_Demand = data.groupby('StockCode')[target_column]
    
    for i in [1,7,14,28]:
        data[f'lag_{i}'] = grouped_Demand.shift(i)
    
    return data

def add_rolling_features(df: pd.DataFrame, target_column: int = TARGET_COLUMN) -> pd.DataFrame:
    """Add rolling features using values before current date"""
    data = df.copy()
    
    prev_demand = data.groupby('StockCode')['demand'].shift(1)
    data['rolling_mean_7'] = (prev_demand.groupby(data['StockCode']).rolling(window=7, min_periods=3).mean().reset_index(level=0, drop=True))
    
    data['rolling_mean_14'] = prev_demand.groupby(data['StockCode']).rolling(window=14, min_periods=7).mean().reset_index(level=0, drop=True)
    
    data['rolling_std_7'] = prev_demand.groupby(data['StockCode']).rolling(window=7, min_periods=3).std().reset_index(level=0, drop=True)
    
    data['rolling_max_7'] = prev_demand.groupby(data['StockCode']).rolling(window=7, min_periods=3).max().reset_index(level=0, drop=True)
    
    return data

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add avergae price of current and previous"""
    data = df.copy()
    
    data['price'] = data['average_price']
    
    data['average_price_7'] = data.groupby('StockCode')['price'].transform(lambda x: x.shift(1).rolling(window=7, min_periods = 3).mean())
    
    return data

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the complete feature table"""
    data =df.copy()
    
    data = data.sort_values(['StockCode', 'Date']).reset_index(drop=True)
    
    data = create_calendar_dates(data)
    data = add_lag_features(data)
    data = add_rolling_features(data)
    data = add_price_features(data)
    
    data = data.dropna(subset= FEATURE_COLUMNS + [TARGET_COLUMN])
    
    return data

def save_features(df:pd.DataFrame, output_path: str) -> None:
    """Successfully save the complete features table"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)

if __name__ == '__main__':
    input_path = 'data/processed/product_day_demand.parquet'
    output_path = 'data/processed/model_features.parquet'
    
    demand = pd.read_parquet(input_path)
    features = create_features(demand)
    save_features(features, output_path)
    
    print(f'Save the feature file with size:{len(features)}')
    print(f'features: {FEATURE_COLUMNS}')