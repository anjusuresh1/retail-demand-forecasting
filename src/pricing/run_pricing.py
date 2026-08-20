from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pricing.elasticity import (
    estimate_all_elasticities,
)
from src.pricing.optimizer import (
    PricingConstraints,
    recommend_price,
)
from src.features.build_features import (
    FEATURE_COLUMNS,
)


FEATURE_FILE = Path(
    "data/processed/model_features.parquet"
)

MODEL_FILE = Path(
    "models/tuned_xgboost.joblib"
)

ELASTICITY_FILE = Path(
    "data/processed/elasticity_estimates.csv"
)

OUTPUT_FILE = Path(
    "data/processed/pricing_recommendations.csv"
)

def load_inputs():
    """Load feature data, model, and elasticity estimates."""
    data = pd.read_parquet(
        FEATURE_FILE
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    model = joblib.load(
        MODEL_FILE
    )

    # elasticity_df = pd.read_csv(
    #     ELASTICITY_FILE
    # )

    return data, model

def create_product_snapshot(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create the most recent row for each product."""
    data = data.sort_values(
        ["StockCode", "Date"]
    )

    snapshot = (
        data.groupby("StockCode")
        .tail(1)
        .copy()
    )

    return snapshot

def predict_base_demand(
    snapshot: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Predict base demand at the current price."""
    result = snapshot.copy()

    result["base_demand"] = model.predict(
        result[FEATURE_COLUMNS]
    )

    result["base_demand"] = np.clip(
        result["base_demand"],
        a_min=0,
        a_max=None,
    )

    return result

def estimate_unit_cost(
    price: float,
    cost_ratio: float = 0.60,
) -> float:
    """Estimate unit cost as a proportion of price."""
    return float(price * cost_ratio)

def generate_recommendations(
    snapshot: pd.DataFrame,
    elasticity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a price recommendation per product."""
    merged = snapshot.merge(
        elasticity_df[
            [
                "StockCode",
                "Elasticity",
                "Estimate_status",
            ]
        ],
        on="StockCode",
        how="left",
    )

    merged["Elasticity"] = (
        merged["Elasticity"]
        .fillna(-1.0)
    )

    recommendations = []

    for _, row in merged.iterrows():
        current_price = float(
            row["average_price"]
        )

        unit_cost = estimate_unit_cost(
            current_price
        )

        constraints = PricingConstraints(
            min_price=current_price * 0.80,
            max_price=current_price * 1.20,
            unit_cost=unit_cost,
            price_step=max(
                current_price * 0.02,
                0.01,
            ),
            minimum_margin=0.10,
        )

        recommendation = recommend_price(
            base_demand=float(
                row["base_demand"]
            ),
            current_price=current_price,
            elasticity=float(
                row["Elasticity"]
            ),
            constraints=constraints,
        )

        recommendation["StockCode"] = (
            row["StockCode"]
        )

        recommendation[
            "Estimate_status"
        ] = row["Estimate_status"]

        recommendation[
            "unit_cost_assumption"
        ] = unit_cost

        recommendations.append(
            recommendation
        )

    return pd.DataFrame(
        recommendations
    )
def save_recommendations(
    recommendations: pd.DataFrame,
) -> None:
    """Save pricing recommendations."""
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def main() -> None:
    data, model = (
        load_inputs()
    )

    elasticity_estimates = (
        estimate_all_elasticities(data)
    )

    elasticity_estimates.to_csv(
        ELASTICITY_FILE,
        index=False,
    )

    snapshot = create_product_snapshot(
        data
    )

    snapshot = predict_base_demand(
        snapshot,
        model,
    )

    recommendations = (
        generate_recommendations(
            snapshot,
            elasticity_estimates,
        )
    )

    save_recommendations(
        recommendations
    )

    print(
        recommendations.head(
            10
        ).to_string(index=False)
    )

    print(
        f"\nSaved recommendations to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()