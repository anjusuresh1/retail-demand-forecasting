from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import (
    FEATURE_COLUMNS,
)
from src.pricing.optimizer import (
    PricingConstraints,
    recommend_price,
)

MODEL_PATH = Path(
    "models/tuned_xgboost.joblib"
)

MODEL_VERSION = "xgboost-day3"

_model = None

def load_model():
    """Load the trained demand model."""
    global _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    _model = joblib.load(
        MODEL_PATH
    )

    return _model

def request_to_dataframe(
    request,
) -> pd.DataFrame:
    """Convert validated request fields to model input."""
    request_data = request.model_dump()

    model_input = pd.DataFrame(
        [request_data]
    )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in model_input.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    model_input = model_input[
        FEATURE_COLUMNS
    ].copy()

    return model_input

def predict_demand(
    request,
) -> float:
    """Generate a non-negative demand prediction."""
    if _model is None:
        load_model()

    model_input = request_to_dataframe(
        request
    )

    prediction = _model.predict(
        model_input
    )[0]

    return max(
        float(prediction),
        0.0,
    )
    
def generate_price_recommendation(
    request,
) -> dict:
    """Generate a constrained price recommendation."""
    constraints = PricingConstraints(
        min_price=request.min_price,
        max_price=request.max_price,
        unit_cost=request.unit_cost,
        price_step=max(
            request.current_price * 0.02,
            0.01,
        ),
        minimum_margin=request.minimum_margin,
    )

    return recommend_price(
        base_demand=request.base_demand,
        current_price=request.current_price,
        elasticity=request.elasticity,
        constraints=constraints,
    )
    
