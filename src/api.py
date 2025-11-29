"""
FastAPI Application for Stock Price Prediction Model Serving
Includes Prometheus metrics integration for monitoring
"""
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Import configuration
from src.config import MLFLOW_TRACKING_URI, MODELS_DIR, DAGSHUB_USERNAME, DAGSHUB_TOKEN

# Initialize FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="MLOps Stock Prediction Model Serving API",
    version="1.0.0"
)

# ==================== Prometheus Metrics ====================
# Request counter
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

# Inference latency histogram
INFERENCE_LATENCY = Histogram(
    'api_inference_latency_seconds',
    'API inference latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Data drift counter
DATA_DRIFT_COUNTER = Counter(
    'data_drift_detected_total',
    'Total number of data drift detections'
)

# ==================== Model Loading ====================
model = None
feature_names = None
model_loaded_at = None

def load_model():
    """Load the trained model from local storage or MLflow"""
    global model, feature_names, model_loaded_at
    
    if model is not None:
        return model
    
    # Try to load from local models directory first
    model_path = Path(MODELS_DIR) / 'stock_model.pkl'
    
    if model_path.exists():
        print(f"Loading model from local path: {model_path}")
        model = joblib.load(model_path)
        model_loaded_at = datetime.now()
        
        # Try to load feature names if available
        feature_importance_path = Path(MODELS_DIR) / 'feature_importance.csv'
        if feature_importance_path.exists():
            feature_df = pd.read_csv(feature_importance_path)
            feature_names = feature_df['feature'].tolist()
        else:
            # If no feature importance file, use default feature names
            # This should match the features created in data_transformation.py
            feature_names = None
        
        print(f"Model loaded successfully at {model_loaded_at}")
        return model
    
    # If local model doesn't exist, try MLflow
    try:
        import mlflow
        import mlflow.sklearn
        
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Set credentials if available
            if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
                os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
                os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
            
            # Get the latest model from MLflow
            print("Attempting to load model from MLflow...")
            # For now, we'll use local model. MLflow model registry integration can be added later
            raise FileNotFoundError("MLflow model loading not yet implemented")
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model not found. Please ensure model is trained and saved at {model_path}"
        )

# Load model at startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")

# ==================== Pydantic Models ====================
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    features: List[float] = Field(..., description="Feature vector for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [150.0, 152.0, 149.0, 151.0, 1000000.0, 0.5, 0.3, 0.2]
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: float = Field(..., description="Predicted stock price")
    model_version: Optional[str] = Field(None, description="Model version/timestamp")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_loaded_at: Optional[str] = None
    timestamp: str

# ==================== API Endpoints ====================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    global model, model_loaded_at
    
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_loaded_at=model_loaded_at.isoformat() if model_loaded_at else None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict stock price based on feature vector
    
    Expected features (in order):
    - open, high, low, close, volume
    - lag features (1h, 2h, 3h, etc.)
    - rolling statistics
    - time-based features
    """
    global model, feature_names
    
    start_time = time.time()
    
    try:
        # Load model if not already loaded
        if model is None:
            load_model()
        
        if model is None:
            REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert features to numpy array
        features_array = np.array(request.features).reshape(1, -1)
        
        # Check for data drift (simplified: check if features are out of expected range)
        # This is a basic implementation - can be enhanced with statistical tests
        if detect_data_drift(features_array):
            DATA_DRIFT_COUNTER.inc()
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        INFERENCE_LATENCY.observe(time.time() - start_time)
        
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version=model_loaded_at.isoformat() if model_loaded_at else None,
            inference_time_ms=round(inference_time, 3)
        )
    
    except ValueError as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='400').inc()
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/metrics', status='200').inc()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ==================== Helper Functions ====================
def detect_data_drift(features: np.ndarray) -> bool:
    """
    Simple data drift detection
    Checks if features are outside reasonable ranges for stock data
    
    Args:
        features: Feature array
        
    Returns:
        bool: True if drift detected
    """
    # Basic checks: prices should be positive, volumes should be positive
    # This is a simplified version - can be enhanced with statistical tests
    if np.any(features < 0):
        return True
    
    # Check for extreme values (prices > 10000 or volumes > 1e10)
    if np.any(features > 1e10):
        return True
    
    return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


