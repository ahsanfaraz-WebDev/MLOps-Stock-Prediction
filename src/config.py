"""
Configuration file for MLOps Stock Prediction Project
Contains all configuration settings for API keys, MLflow, and Dagshub
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Alpha Vantage API Configuration ====================
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'AAPL')  # Default to Apple stock

# ==================== Dagshub Configuration ====================
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', '')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', 'MLOps-Stock-Prediction')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')

# MLflow Tracking URI (Dagshub)
# Format: https://dagshub.com/USERNAME/REPO_NAME.mlflow
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow" if DAGSHUB_USERNAME else ''
)

# ==================== Data Paths ====================
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'

# ==================== Model Configuration ====================
MODEL_NAME = 'stock_prediction_model'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==================== Data Quality Thresholds ====================
NULL_THRESHOLD = 0.01  # 1% null values allowed

# ==================== Monitoring Configuration ====================
PROMETHEUS_PORT = 8000
GRAFANA_PORT = 3000

# Test change for complete workflow verification - PR #1, #2, #3

