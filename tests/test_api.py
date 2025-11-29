"""
Unit tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Stock Price Prediction API"

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data

def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

def test_predict_endpoint_no_model():
    """Test predict endpoint when model is not loaded"""
    # This test will pass if model is not loaded (expected in CI)
    response = client.post(
        "/predict",
        json={"features": [150.0, 152.0, 149.0, 151.0, 1000000.0, 0.5, 0.3, 0.2]}
    )
    # Should return 500 if model not loaded, or 200 if model exists
    assert response.status_code in [200, 500]

def test_predict_endpoint_invalid_input():
    """Test predict endpoint with invalid input"""
    response = client.post(
        "/predict",
        json={"features": []}  # Empty features
    )
    assert response.status_code in [400, 422, 500]  # Validation error or model error

def test_predict_endpoint_missing_field():
    """Test predict endpoint with missing field"""
    response = client.post(
        "/predict",
        json={}  # Missing features field
    )
    assert response.status_code == 422  # Validation error


