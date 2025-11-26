"""
Model Training Script for Stock Prediction Pipeline
Trains ML model and logs everything to MLflow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime
import os
import sys
from pathlib import Path

# Import configuration
from src.config import (
    MLFLOW_TRACKING_URI,
    DAGSHUB_USERNAME,
    DAGSHUB_TOKEN,
    DAGSHUB_REPO,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE
)


def train_model(data_path):
    """
    Train stock price prediction model
    
    Args:
        data_path: Path to processed CSV file
    
    Returns:
        str: Path to saved model file
    """
    print(f"Training model with data from: {data_path}")
    
    # Load processed data
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} rows from {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
    # Separate features and target
    if 'target' not in df.columns:
        raise Exception("Target column 'target' not found in data")
    
    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols]
    y = df['target']
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples after cleaning: {len(X)}")
    
    if len(X) == 0:
        raise Exception("No valid data after cleaning")
    
    # Train-test split (no shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Configure MLflow
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Set credentials if provided
    if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    # Start MLflow run
    run_name = f"stock_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        with mlflow.start_run(run_name=run_name):
            # Hyperparameters
            n_estimators = 100
            max_depth = 10
            min_samples_split = 2
            min_samples_leaf = 1
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            
            # Train model
            print("Training Random Forest model...")
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            print("✓ Model trained successfully")
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            
            # Print metrics
            print(f"\n{'='*60}")
            print(f"Model Performance:")
            print(f"{'='*60}")
            print(f"Train RMSE: {train_rmse:.4f}")
            print(f"Test RMSE:  {test_rmse:.4f}")
            print(f"Train MAE:  {train_mae:.4f}")
            print(f"Test MAE:   {test_mae:.4f}")
            print(f"Train R²:   {train_r2:.4f}")
            print(f"Test R²:    {test_r2:.4f}")
            print(f"{'='*60}\n")
            
            # Save model locally
            model_dir = Path(MODELS_DIR)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'stock_model.pkl'
            joblib.dump(model, model_path)
            print(f"✓ Model saved to: {model_path}")
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(str(model_path))
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_path = model_dir / 'feature_importance.csv'
            feature_importance.to_csv(feature_importance_path, index=False)
            mlflow.log_artifact(str(feature_importance_path))
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
            
            print(f"\n✓ MLflow run completed!")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
            
            return str(model_path)
            
    except Exception as e:
        print(f"Error in MLflow tracking: {str(e)}")
        # Still save model locally even if MLflow fails
        model_dir = Path(MODELS_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'stock_model.pkl'
        joblib.dump(model, model_path)
        print(f"Model saved locally despite MLflow error: {model_path}")
        raise


if __name__ == "__main__":
    """
    Allow script to be run directly from command line
    Usage: python train.py <data_path>
    """
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_path>")
        print("Example: python train.py data/processed/stock_data_processed_20251126_192923.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    try:
        model_path = train_model(data_path)
        print(f"\n✓ Training completed successfully!")
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

