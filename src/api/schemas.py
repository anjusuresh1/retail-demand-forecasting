from typing import List
from pydantic import BaseModel, Field

class DemandPredictionRequest(BaseModel):
    lag_1: float = Field(
        ...,
        ge=0,
        description='Demand from one day ago')
    
    lag_7: float = Field(
        ...,
        ge=0,
        description='Demand from seven days ago'
    )
    
    lag_14: float = Field(
        ...,
        ge=0,
        description="Demand from fourteen days ago",
    )

    lag_28: float = Field(
        ...,
        ge=0,
        description="Demand from twenty-eight days ago",
    )

    rolling_mean_7: float = Field(
        ...,
        ge=0,
        description="Previous seven-day average demand",
    )

    rolling_mean_14: float = Field(
        ...,
        ge=0,
        description="Previous fourteen-day average demand",
    )

    rolling_std_7: float = Field(
        ...,
        ge=0,
        description="Previous seven-day demand standard deviation",
    )

    rolling_max_7: float = Field(
        ...,
        ge=0,
        description="Maximum demand in the previous seven days",
    )

    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Monday is 0 and Sunday is 6",
    )

    month: int = Field(
        ...,
        ge=1,
        le=12,
    )

    day_of_month: int = Field(
        ...,
        ge=1,
        le=31,
    )

    week_of_year: int = Field(
        ...,
        ge=1,
        le=53,
    )

    quarter: int = Field(
        ...,
        ge=1,
        le=4,
    )

    is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
    )

    price: float = Field(
        ...,
        gt=0,
        description="Current product price",
    )

    average_price_7: float = Field(
        ...,
        gt=0,
        description="Previous seven-day average price",
    )

class DemandPredictionResponse(BaseModel):
    predicted_demand: float
    model_version: str
    latency_ms: float
    
class PriceRecommendationRequest(BaseModel):
    """Input data for price optimization."""

    base_demand: float = Field(
        ...,
        ge=0,
        description="Expected demand at the current price",
    )

    current_price: float = Field(
        ...,
        gt=0,
    )

    elasticity: float = Field(
        ...,
        le=0,
        description="Expected non-positive price elasticity",
    )

    unit_cost: float = Field(
        ...,
        gt=0,
    )

    min_price: float = Field(
        ...,
        gt=0,
    )

    max_price: float = Field(
        ...,
        gt=0,
    )

    minimum_margin: float = Field(
        0.10,
        ge=0,
        le=1,
    )
    
class PriceRecommendationResponse(BaseModel):
    """Output returned by the pricing endpoint."""

    current_price: float
    recommended_price: float
    base_demand: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    expected_margin: float
    elasticity: float
    candidate_count: int
    
