from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 

MIN_OBSERVATION = 20
MIN_PRICE_VARIATION = 0.01
DEFAULT_ELASTICITY = -1

def prepare_elasticity_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        'StockCode',
        'demand',
        'average_price'
    }
    
    missing_columns = required_cols.difference(df.columns)
    
    if missing_columns:
        raise ValueError(
            print(f'Missing columns : {missing_columns}')
        )
    
    data = df[['StockCode', 'demand', 'average_price']].copy()
    
    data = data.rename(columns = {'average_price': 'price'})
    data = data[(data['demand'] > 0) & (data['price'] > 0) ]
    
    data['log_price'] = np.log(data['price'])
    
    data['log_demand'] = np.log(data['demand'])
    
    return data

def estimate_product_elasticity(df: pd.DataFrame ) -> tuple[float, str]:
    if len(df) < MIN_OBSERVATION:
        return DEFAULT_ELASTICITY , 'fallback_low_observations'
    
    min_price = df['price'].min()
    max_price = df['price'].max()
    
    price_variation = (max_price - min_price) / max(min_price, 1e-8)
    
    if price_variation < MIN_PRICE_VARIATION:
        return DEFAULT_ELASTICITY, 'fallback_low_price_variation'
    
    X = df[['log_price']]
    y = df[['log_demand']]
    
    model = LinearRegression()
    model.fit(X, y)
    
    elasticity = model.coef_[0]
    if not np.isfinite(elasticity):
        return elasticity, 'fallback_nonfinite_elasticity'      
        
    return elasticity, 'estimated'

def estimate_all_elasticities(df: pd.DataFrame) -> pd.DataFrame:
    data = prepare_elasticity_data(df)
    
    estimates = []
    
    for stock_code, product_data in data.groupby('StockCode'):
        elasticity, status = estimate_product_elasticity(product_data)
        
        estimates.append({
            'StockCode': stock_code,
            'Elasticity': elasticity,
            'Estimate_status': status,
            'Observations': len(product_data),
            'Price_min': product_data['price'].min(),
            'Price_max': product_data['price'].max()
        })
        
    return pd.DataFrame(estimates)

def save_elasticities(df: pd.DataFrame, output_file: str) -> None:
    output_path = Path(output_file)
    
    output_path.parent.mkdir(parents = True, exist_ok = True)
    
    df.to_csv(output_path, index = False)
    
