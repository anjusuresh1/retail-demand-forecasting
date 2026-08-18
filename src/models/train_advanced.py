from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost

from xgboost import XGBRegressor
import pandas as pd
import numpy as np

from src.features.build_features import FEATURE_COLUMNS
from src.models.metrics import regression_metrics

TARGET_COLUMN = 'demand'

INPUT_FILE = Path('data/processed/model_features.parquet')

OUTPUT_DIRECTORY = Path('models')

OUTPUT_FILE = Path('data/processed/model_comparison_day3.csv')

def load_features(input_path: Path) -> pd.DataFrame:
    data = pd.read_parquet(input_path)
    required_columns = set(FEATURE_COLUMNS)
    
    required_columns.update(
        {
            TARGET_COLUMN,
            'Date',
            'StockCode'
        }
    )
    
    missing_columns = (required_columns.difference(data.columns))
    
    if missing_columns:
        raise ValueError(
            print(f'Missing columns are present of rows: {len(missing_columns)}')
        )
        
    data = data.dropna(subset = FEATURE_COLUMNS + [TARGET_COLUMN, 'Date']).copy()
    
    return data.sort_values(['Date', 'StockCode']).reset_index(drop = True)

def time_split(data: pd.DataFrame, validation_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Split testing and validation data through time split
    end_date = data['Date'].max()
    
    cutoff_date = (end_date - pd.Timedelta(validation_days - 1))
    
    train = data[data['Date'] < cutoff_date]
    
    test = data[data['Date'] >= cutoff_date]
    
    return train, test

def prepare_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    
    return X,y

def create_xgboost_model(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8
)-> XGBRegressor:
    return XGBRegressor(
        objective = 'reg:squarederror',
        n_estimators= n_estimators,
        max_depth= max_depth,
        learning_rate= learning_rate,
        min_child_weight = min_child_weight,
        subsample= subsample,
        colsample_bytree= colsample_bytree,
        reg_alpha = 0.0,
        reg_lambda = 1.0,
        random_state = 42,
        n_jobs = 4
    )

def calculate_metrics(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray
) -> dict:
    predictions = np.clip(
        predictions,
        a_min = 0,
        a_max = None
    )
    
    metrics = regression_metrics(y_true , predictions)
    metrics['model'] = model_name
    
    return metrics

def train_and_evaluate(
    model_name: str,
    model: XGBRegressor,
    train: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[XGBRegressor, dict]:
    
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)
    
    with mlflow.start_run(run_name= model_name):
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        metrics = calculate_metrics(model_name, y_test, predictions)
        
        mlflow.log_params(model.get_params())
        
        mlflow.log_metrics({
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'WAPE': metrics['WAPE']
        })
        
        mlflow.xgboost.log_model(model, artifact_path = 'model')
    
    return model , metrics

def save_feature_importance(model: XGBRegressor, output_file) -> pd.DataFrame:
    importance = pd.DataFrame({
        'features' : FEATURE_COLUMNS,
        'importance': model.feature_importances_
    })
    
    importance.sort_values('importance', ascending = False)
    
    output_file.parent.mkdir(parents=True, exist_ok= True)
    
    importance.to_csv(output_file,index=False)
    
    return importance
        
def plot_top_features(output_file: Path, importance: pd.DataFrame) -> None:
    top_features = importance.head(15)
    top_features = top_features.sort_values('importance')
    
    plt.figure(figsize=(10,7))
    
    plt.barh(top_features['features'], top_features['importance'])
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Top XGBoost Features')
    plt.tight_layout()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=150)
    
    plt.close()
    
def main() -> None:
    features = load_features(INPUT_FILE)
    
    train, test = time_split(features, validation_days=28)
    
    mlflow.set_experiment('Retail-demand-forecasting')
    
    model_configs = {
        'xg_baseline':{
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'xg_regularized':{
            'n_estimators': 500,
            'max_depth': 4,
            'learning_rate': 0.03,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'xg_shallow':{
            'n_estimators': 500,
            'max_depth': 3,
            'learning_rate': 0.03,
            'min_child_weight': 3,
            'subsample': 0.09,
            'colsample_bytree': 0.09
        }
    }
    
    results = []
    trained_models = {}
    
    for model_name, configs in (model_configs.items()):
        model = create_xgboost_model(**configs)
        
        trained_model, metrics = train_and_evaluate(model_name, model, train, test)
        
        results.append(metrics)
        trained_models[model_name] = trained_model
        
    results_df = pd.DataFrame(results)
    results_df = results_df[[
        'model',
        'MAE',
        'RMSE',
        'MAPE',
        'WAPE'
    ]]
    
    results_df = results_df.sort_values('WAPE')
    
    OUTPUT_FILE.parent.mkdir(parents= True, exist_ok = True)
    
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    best_model_name = results_df.iloc[0]['model']
    
    best_model = trained_models[best_model_name]
    
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    model_file = (OUTPUT_DIRECTORY/'tuned_xgboost.joblib')
    
    joblib.dump(model, model_file)
    
    important_file = (OUTPUT_DIRECTORY/'important_features.csv')
    
    importance= save_feature_importance(best_model, important_file)
    
    plot_file = (OUTPUT_DIRECTORY/'important_features.png')
    
    plot_top_features(plot_file, importance)
    
    print('Model comparison')
    print(results_df.to_string(index=False))
    
    print(f'Best model: {best_model_name}')
    print(f'Best model saved to file {model_file}')
    
    print(f'Saved important features to file {important_file}')
    
    print(f'Saved important feature plot to file {plot_file}')
    
if __name__ == '__main__':
    main()
        
