"""
Unit tests for ETL pipeline
"""

import pandas as pd
import numpy as np
import pytest
from src.data.etl import data_validation, time_based_features


def test_validate_data_removes_missing():
    """Test that validation removes rows with missing essential fields"""
    df = pd.DataFrame({
        'Invoice': ['A', 'B', None, 'D'],
        'StockCode': ['X', 'Y', 'Z', None],
        'Quantity': [1, 2, 3, 4],
        'Price': [10, 20, 30, 40],
        'Customer ID': [1,2,3,4],
        'InvoiceDate': ["2023-01-01 08:30:00",
    "2023-01-01 09:15:00",
    "2023-01-01 10:45:00","2023-01-01 10:55:00"]
    })
    
    df_clean = data_validation(df)
    
    # Should remove rows with missing InvoiceNo or StockCode
    assert len(df_clean) < len(df)
    assert df_clean['Invoice'].isnull().sum() == 0
    assert df_clean['StockCode'].isnull().sum() == 0


def test_validate_data_removes_negative_quantity():
    """Test that validation removes negative quantities (returns)"""
    df = pd.DataFrame({
        'Invoice': ['A', 'B', 'C'],
        'StockCode': ['X', 'Y', 'Z'],
        'Quantity': [5, -3, 10],  # One return
        'Price': [10, 20, 30],
        'Customer ID': [1,2,3],
        'InvoiceDate': ["2023-01-01 08:30:00",
    "2023-01-01 09:15:00",
    "2023-01-01 10:45:00",]
    })
    
    df_clean = data_validation(df)
    
    # Should remove the return
    assert len(df_clean) == 1
    assert all(df_clean['Quantity'] > 0)


def test_add_time_features():
    """Test that time features are added correctly"""
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
    })
    
    df_features = time_based_features(df)
    
    # Check that time features are added
    assert 'Year' in df_features.columns
    assert 'Month' in df_features.columns
    assert 'DayOfWeek' in df_features.columns
    assert 'IsWeekend' in df_features.columns
    
    # Verify values
    assert df_features['Year'].iloc[0] == 2024
    assert df_features['Month'].iloc[0] == 1