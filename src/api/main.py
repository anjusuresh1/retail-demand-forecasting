import time
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    DemandPredictionRequest,
    DemandPredictionResponse,
    PriceRecommendationRequest,
    PriceRecommendationResponse,
)
from src.api.services import (
    MODEL_VERSION,
    is_model_loaded,
    generate_price_recommendation,
    load_model,
    predict_demand,
)

app = FastAPI(
    title = 'Retail Demand Forecasting API',
    description = ('API for demand prediction and '
                   'price recommendations'),
    version = '1.0.0'
)

@app.on_event('startup')
def start_event():
    load_model()
    
@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': is_model_loaded(),
        'version': MODEL_VERSION
    }
    
@app.post(
    '/v1/predict/demand',
    response_model = DemandPredictionResponse
)
def predict_demand_endpoint(
    request: DemandPredictionRequest
):
    start_time = time.perf_counter()
    
    try:
        prediction = predict_demand(request)
    except Exception as error:
        raise HTTPException(
            status_code = 500,
            detail = f'Prediction failed {error}'
        ) from error
    
    latency_ms = time.perf_counter() - start_time
    
    return DemandPredictionResponse(
        predicted_demand= prediction,
        model_version=(
            f'{MODEL_VERSION};'
        ),
        latency_ms=float(latency_ms)
    )
    
@app.post(
    '/v1/recommend/price',
    response_model = PriceRecommendationResponse
)
def recommend_price_endpoint(
    request: PriceRecommendationRequest
):
    try:
        recommendation = (
            generate_price_recommendation(request)
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            details=str(error)
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            details= ('Pricing recommendation failed'
                      f'{error}')
        ) from error
    
    return PriceRecommendationResponse(
        **recommendation
    )

        