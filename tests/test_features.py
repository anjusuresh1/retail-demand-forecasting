import pandas as pd

from src.features.build_features import (
    create_calendar_dates,
    add_lag_features,
    create_features
)

def sample_data() -> pd.DataFrame:
    dates = pd.date_range(
        start="2024-01-01",
        periods=40,
        freq='D'
    )
    
    return pd.DataFrame(
        {
            "StockCode": ["A"] * 40,
            "Date": dates,
            "demand": list(range(1, 41)),
            "revenue": [10.0] * 40,
            "average_price": [10.0] * 40,
            "transaction_count": [1] * 40
        }
    )

def test_calendar_features():
    data = sample_data()
    result = create_calendar_dates(data)
    
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    assert 'is_weekend' in result.columns
    
def test_lag_features():
    data = sample_data()
    result = add_lag_features(data)
    
    assert result['lag_1'].iloc[1] == 1
    assert result['lag_7'].iloc[7] == 1
    
def test_create_features_missing_null():
    data = sample_data()
    result = create_features(data)
    
    assert not result.isnull().any().any()
    assert 'rolling_mean_7' in result.columns