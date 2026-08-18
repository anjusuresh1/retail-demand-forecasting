import joblib
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS

def test_saved_model_can_predict():
    model = joblib.load('models/tuned_xgboost.joblib')
    
    features = pd.read_parquet('data/processed/model_features.parquet')
    
    sample_features = features[FEATURE_COLUMNS].head(5)
    
    predictions = model.predict(sample_features)
    
    assert len(predictions) == 5
    assert all(predictions > 0)