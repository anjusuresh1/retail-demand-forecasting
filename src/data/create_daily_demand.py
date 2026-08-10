from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def load_transactions(input_path: str) -> pd.DataFrame:
    """Load cleaned transactions Data"""
    logger.info('Loading cleaned transactions data from path %s', input_path)
    
    df = pd.read_parquet(input_path)
    
    required_columns = {'StockCode', 'Date', 'Quantity', 'Price'}
    missing_columns = required_columns.difference(df.columns)
    
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')
    
    return df

def create_product_day_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate based of daily product demand"""
    logger.info('Creating product day demand table')
    
    data = df.copy()
    
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data['Revenue'] = data['Quantity'] * data['Price']
    
    daily = data.groupby(['StockCode', 'Date'], as_index = False).agg(
        demand = ('Quantity', 'sum'),
        revenue = ('Revenue', 'sum'),
        average_price = ('Price', 'mean'),
        transaction_count = ('Quantity', 'size')
    )
    
    daily = daily.sort_values(['StockCode', 'Date'])
    
    logger.info('Daily demand table created with %d rows', len(daily))
    return daily

def save_demand_table(df: pd.DataFrame, output_path: str) -> None:
    """Save the day demand table"""
    logger.info('Saving daily demand table to path %s', output_path)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents = True, exist_ok = True)
    
    df.to_parquet(output_file, index = False)
    
    logger.info('Daily demand table saved successfully to %s', output_file)

def complete_product_dates(daily: pd.DataFrame, top_n_products: int = 50) -> pd.DataFrame:
    """Complete date range for top n products"""
    data = daily.copy()
    
    top_products = data.groupby('StockCode')['Revenue'].sum().nlargest(top_n_products).reset_index()
    data = data[data['StockCode'].isin(top_products['StockCode'])].copy()
    
    start_date = data['Date'].min()
    end_date = data['Date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    product_date_combo_index = pd.MultiIndex.from_product([top_products['StockCode'], all_dates], names=['StockCode', 'Date'])
    
    data = data.set_index(['StockCode', 'Date']).reindex(product_date_combo_index).reset_index()
    
    data[['demand', 'revenue', 'transaction_count']] = data[['demand', 'revenue', 'transaction_count']].fillna(0)
    
    data['average_price'] = data.groupby('StockCode')['average_price'].transform(lambda x: x.ffill().bfill())
    
    return data.sort_values(['StockCode', 'Date'])

if __name__ == "__main__":
    input_path = 'data/processed/sales_daily.parquet'
    output_path = 'data/processed/product_day_demand.parquet'
    
    transactions_df = load_transactions(input_path)
    daily_demand_df = create_product_day_demand(transactions_df)
    daily_demand_df = complete_product_dates(daily_demand_df, top_n_products = 50)
    save_demand_table(daily_demand_df, output_path)
    
 