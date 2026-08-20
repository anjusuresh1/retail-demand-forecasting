from dataclasses import dataclass
import numpy as np

@dataclass
class PricingConstraints:
    min_price: float
    max_price: float 
    unit_cost: float
    price_step: float = 0.50
    minimum_margin: float = 0.10

def generate_candidate_price(current_price: float, constraints: PricingConstraints, adjustment: float =0.2) -> np.ndarray:
    lowerBound = max(constraints.min_price, current_price * (1 - adjustment))
    
    upperbound = min(constraints.max_price, current_price * (1 + adjustment))
    
    prices = np.arange(lowerBound, upperbound + constraints.price_step, constraints.price_step)
    
    prices = np.round(prices, decimals = 2)
    
    margin_condition = (prices - constraints.unit_cost) / prices >= constraints.minimum_margin
    
    return prices[margin_condition]

def demand_at_price (
    base_demand: float,
    current_price: float,
    candidate_price: float,
    elasticity: float
) -> float:
    price_ratio = (candidate_price / max(current_price, 1e-8))
    
    adjusted_demand = base_demand * price_ratio ** elasticity
    
    return float(max(adjusted_demand, 0.0))

def calculate_outcomes(
    price: float,
    expected_demand: float,
    unit_cost: float,
) -> dict:
    """Calculate revenue and profit for a price."""
    revenue = price * expected_demand

    profit = (
        price - unit_cost
    ) * expected_demand

    margin = (
        profit / revenue
        if revenue > 0
        else 0.0
    )

    return {
        "price": float(price),
        "expected_demand": float(
            expected_demand
        ),
        "expected_revenue": float(
            revenue
        ),
        "expected_profit": float(
            profit
        ),
        "expected_margin": float(
            margin
        ),
    }
    
def recommend_price(
    base_demand: float,
    current_price: float,
    elasticity: float,
    constraints: PricingConstraints,
) -> dict:
    """Recommend the price with maximum expected profit."""
    candidate_prices = (
        generate_candidate_price(
            current_price,
            constraints,
        )
    )

    if len(candidate_prices) == 0:
        raise ValueError(
            "No candidate prices satisfy "
            "the pricing constraints."
        )

    outcomes = []

    for candidate_price in candidate_prices:
        expected_demand = demand_at_price(
            base_demand=base_demand,
            current_price=current_price,
            candidate_price=candidate_price,
            elasticity=elasticity,
        )

        outcome = calculate_outcomes(
            price=candidate_price,
            expected_demand=expected_demand,
            unit_cost=constraints.unit_cost,
        )

        outcomes.append(outcome)

    best_outcome = max(
        outcomes,
        key=lambda item: item[
            "expected_profit"
        ],
    )

    return {
        "current_price": current_price,
        "base_demand": base_demand,
        "elasticity": elasticity,
        "recommended_price": best_outcome[
            "price"
        ],
        "expected_demand": best_outcome[
            "expected_demand"
        ],
        "expected_revenue": best_outcome[
            "expected_revenue"
        ],
        "expected_profit": best_outcome[
            "expected_profit"
        ],
        "expected_margin": best_outcome[
            "expected_margin"
        ],
        "candidate_count": len(
            candidate_prices
        ),
    }