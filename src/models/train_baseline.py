import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


from src.features.build_features import FEATURE_COLUMNS
from src.models.metrics import regression_metrics

def time_split(df: pd.DataFrame, test_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data chronologically"""
    data = df.copy()
    data = data.sort_values('Date')
    
    cutoff_date = (data['Date'].max() - pd.Timedelta(days=test_days - 1))
    
    train_data = data[data['Date'] < cutoff_date].copy()
    test_data = data[data['Date'] >= cutoff_date].copy()
    
    return train_data, test_data

def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Seperate features and target"""
    X = df[FEATURE_COLUMNS].copy()
    y = df['demand'].copy()
    
    return X,y

def evaluate_predictions(
    model_name:str,
    y_true: pd.Series,
    predictions: pd.Series
) -> dict:
    """Evaluate Model performance and attach model name"""
    metrics = regression_metrics(y_true, predictions)
    metrics['model'] = model_name
    
    return metrics

def evaluate_naive_weekly(df:pd.DataFrame, test: pd.DataFrame) -> dict:
    """Evaluate a forecast equal to 7 day demand"""
    actual_data = df.sort_values(['StockCode','Date']).copy()
    actual_data['naive_weekly_prediction'] = actual_data.groupby('StockCode')['demand'].shift(7)
    
    test_data = actual_data[actual_data['Date'].isin(test['Date'])].copy()
    
    test_data = test_data.dropna(subset=['naive_weekly_prediction'])
    
    metrics = regression_metrics(test_data['demand'], test_data['naive_weekly_prediction'])
    
    metrics['model'] = 'NaiveWeekly'
    
    return metrics

def train_ridge(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train and evaluate a regularized linear model"""
    X_train, y_train = prepare_xy(train)
    x_test, y_test = prepare_xy(test)
    
    model = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha = 1.0))
        ]
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(x_test)
    predictions= predictions.clip(min=0)
    
    return evaluate_predictions(
        'RidgeRegression',
        y_test,
        predictions
    )
    
def train_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train and evaluate a XGBoost Rgression Model"""
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators = 300,
        max_depth = 6,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state=42,
        n_jobs= 4
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    predictions = predictions.clip(min=0)
    
    return evaluate_predictions('XGBoost', y_test, predictions)

def main() -> None:
    input_file = 'data/processed/model_features.parquet'
    
    data = pd.read_parquet(input_file)
    data['Date'] = pd.to_datetime(data['Date'])
    
    train, test = time_split(data, test_days = 28)
    
    results =[]
    
    results.append(evaluate_naive_weekly(data, test))
    results.append(train_ridge(train, test))
    results.append(train_xgboost(train, test))
    
    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'MAE', 'RMSE', 'MAPE', 'WAPE']]
    
    print('\nModel comparison')
    print(results_df.to_string(index=False))
    
    results_df.to_csv('data/processed/baseline_results.csv', index=False)
    
if __name__ == '__main__':
    main()