"""
Model Training Script for Stock Prediction Pipeline
Trains ML model and logs everything to MLflow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime
import os
import sys
from pathlib import Path
import warnings

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


def get_adaptive_hyperparameters(n_samples, n_features):
    """
    Get hyperparameters adapted to dataset size to prevent overfitting
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
    
    Returns:
        dict: Hyperparameters for RandomForest
    """
    # For small datasets (< 100 samples), use very conservative settings
    if n_samples < 100:
        return {
            'n_estimators': 30,  # Reduced from 100
            'max_depth': 5,      # Reduced from 10
            'min_samples_split': 10,  # Increased from 2
            'min_samples_leaf': 5,    # Increased from 1
            'max_features': 'sqrt',   # Use sqrt of features per tree
            'max_samples': 0.8        # Use 80% of samples per tree
        }
    # For medium datasets (100-500 samples)
    elif n_samples < 500:
        return {
            'n_estimators': 50,
            'max_depth': 7,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'max_samples': 0.85
        }
    # For larger datasets (500+ samples)
    else:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'max_samples': 0.9
        }


def select_features(X_train, y_train, X_test, n_features_to_select=None):
    """
    Select top features using statistical tests to reduce overfitting
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        n_features_to_select: Number of features to select (auto if None)
    
    Returns:
        tuple: (X_train_selected, X_test_selected, selected_feature_names)
    """
    n_samples, n_features = X_train.shape
    
    # Determine number of features to select
    if n_features_to_select is None:
        # Rule of thumb: max features = min(n_samples/10, n_features/2, 20)
        n_features_to_select = min(max(int(n_samples / 10), 5), int(n_features / 2), 20)
    
    # Don't select if we already have few features
    if n_features_to_select >= n_features:
        return X_train, X_test, X_train.columns.tolist()
    
    print(f"  Selecting top {n_features_to_select} features from {n_features} total...")
    
    # Use SelectKBest with f_regression
    selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_feature_names = X_train.columns[selected_mask].tolist()
    
    print(f"  ✓ Selected {len(selected_feature_names)} features")
    
    # Convert back to DataFrame for easier handling
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_feature_names, index=X_train.index)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_feature_names, index=X_test.index)
    
    return X_train_selected, X_test_selected, selected_feature_names


def train_model(data_path):
    """
    Train stock price prediction model with adaptive hyperparameters
    
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
    
    # Warn if dataset is very small
    if len(X) < 50:
        warnings.warn(
            f"⚠️  Very small dataset ({len(X)} samples). Model performance may be poor. "
            "Consider collecting more data.",
            UserWarning
        )
    
    # Train-test split (no shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Feature selection to reduce overfitting
    print("\n" + "="*60)
    print("Feature Selection")
    print("="*60)
    X_train_selected, X_test_selected, selected_features = select_features(
        X_train, y_train, X_test
    )
    
    # Get adaptive hyperparameters based on dataset size
    hyperparams = get_adaptive_hyperparameters(len(X_train_selected), len(selected_features))
    print(f"\nAdaptive Hyperparameters (for {len(X_train_selected)} samples, {len(selected_features)} features):")
    print(f"  n_estimators: {hyperparams['n_estimators']}")
    print(f"  max_depth: {hyperparams['max_depth']}")
    print(f"  min_samples_split: {hyperparams['min_samples_split']}")
    print(f"  min_samples_leaf: {hyperparams['min_samples_leaf']}")
    print(f"  max_features: {hyperparams['max_features']}")
    print(f"  max_samples: {hyperparams['max_samples']}")
    
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
            # Log parameters
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", hyperparams['n_estimators'])
            mlflow.log_param("max_depth", hyperparams['max_depth'])
            mlflow.log_param("min_samples_split", hyperparams['min_samples_split'])
            mlflow.log_param("min_samples_leaf", hyperparams['min_samples_leaf'])
            mlflow.log_param("max_features", str(hyperparams['max_features']))
            mlflow.log_param("max_samples", hyperparams['max_samples'])
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("n_features_original", len(feature_cols))
            mlflow.log_param("n_features_selected", len(selected_features))
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_train", len(X_train_selected))
            mlflow.log_param("n_test", len(X_test_selected))
            mlflow.log_param("feature_selection", "SelectKBest_f_regression")
            
            # Train model with adaptive hyperparameters
            print("\n" + "="*60)
            print("Model Training")
            print("="*60)
            print("Training Random Forest model...")
            model = RandomForestRegressor(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                max_features=hyperparams['max_features'],
                max_samples=hyperparams['max_samples'],
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(X_train_selected, y_train)
            print("✓ Model trained successfully")
            
            # Make predictions
            y_pred_train = model.predict(X_train_selected)
            y_pred_test = model.predict(X_test_selected)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Calculate overfitting metrics
            rmse_gap = test_rmse - train_rmse
            r2_gap = train_r2 - test_r2
            
            # Calculate baseline metrics (mean prediction)
            train_mean = y_train.mean()
            test_mean = y_test.mean()
            train_baseline_rmse = np.sqrt(mean_squared_error(y_train, [train_mean] * len(y_train)))
            test_baseline_rmse = np.sqrt(mean_squared_error(y_test, [test_mean] * len(y_test)))
            train_std = y_train.std()
            test_std = y_test.std()
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("rmse_gap", rmse_gap)
            mlflow.log_metric("r2_gap", r2_gap)
            mlflow.log_metric("train_baseline_rmse", train_baseline_rmse)
            mlflow.log_metric("test_baseline_rmse", test_baseline_rmse)
            mlflow.log_metric("train_std", train_std)
            mlflow.log_metric("test_std", test_std)
            
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
            print(f"\nBaseline Comparison (Mean Prediction):")
            print(f"Train Baseline RMSE: {train_baseline_rmse:.4f}")
            print(f"Test Baseline RMSE:  {test_baseline_rmse:.4f}")
            print(f"Train Std: {train_std:.4f}")
            print(f"Test Std:  {test_std:.4f}")
            print(f"\nModel vs Baseline:")
            print(f"Train Improvement: {((train_baseline_rmse - train_rmse) / train_baseline_rmse * 100):.2f}%")
            print(f"Test Improvement:  {((test_baseline_rmse - test_rmse) / test_baseline_rmse * 100):.2f}%")
            print(f"{'='*60}")
            
            # Validation checks and warnings
            print(f"\n{'='*60}")
            print(f"Model Validation:")
            print(f"{'='*60}")
            
            # Check for negative R²
            if test_r2 < 0:
                # Check if test set has very low variance (which can cause negative R² even with good RMSE)
                if test_std < train_std * 0.5:
                    warnings.warn(
                        f"⚠️  WARNING: Test R² is negative ({test_r2:.4f}), but test set has low variance "
                        f"(std={test_std:.4f} vs train std={train_std:.4f}). "
                        f"Test RMSE ({test_rmse:.4f}) is actually better than train RMSE ({train_rmse:.4f}). "
                        "This may indicate the test set is easier to predict. Consider collecting more data.",
                        UserWarning
                    )
                    print(f"⚠️  Test R² is negative: {test_r2:.4f}")
                    print(f"   Note: Test RMSE ({test_rmse:.4f}) < Train RMSE ({train_rmse:.4f})")
                    print(f"   Test set variance is low (std={test_std:.4f}), which can cause negative R²")
                else:
                    warnings.warn(
                        f"⚠️  WARNING: Test R² is negative ({test_r2:.4f}). "
                        "Model performs worse than predicting the mean. "
                        "This indicates severe overfitting or data issues.",
                        UserWarning
                    )
                    print(f"⚠️  Test R² is negative: {test_r2:.4f}")
            elif test_r2 < 0.3:
                warnings.warn(
                    f"⚠️  WARNING: Test R² is low ({test_r2:.4f}). "
                    "Model performance is poor. Consider collecting more data.",
                    UserWarning
                )
                print(f"⚠️  Test R² is low: {test_r2:.4f}")
            else:
                print(f"✓ Test R² is acceptable: {test_r2:.4f}")
            
            # Check for overfitting (large gap between train and test)
            if rmse_gap > train_rmse * 2:  # Test RMSE is more than 3x train RMSE
                warnings.warn(
                    f"⚠️  WARNING: Large overfitting detected. "
                    f"RMSE gap: {rmse_gap:.4f} (Test RMSE is {test_rmse/train_rmse:.2f}x train RMSE)",
                    UserWarning
                )
                print(f"⚠️  Overfitting detected: RMSE gap = {rmse_gap:.4f}")
            elif r2_gap > 0.3:  # R² gap > 0.3
                warnings.warn(
                    f"⚠️  WARNING: Moderate overfitting detected. "
                    f"R² gap: {r2_gap:.4f}",
                    UserWarning
                )
                print(f"⚠️  Overfitting detected: R² gap = {r2_gap:.4f}")
            else:
                print(f"✓ Overfitting is acceptable: R² gap = {r2_gap:.4f}")
            
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
            
            # Log feature importance (only for selected features)
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_path = model_dir / 'feature_importance.csv'
            feature_importance.to_csv(feature_importance_path, index=False)
            mlflow.log_artifact(str(feature_importance_path))
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
            
            # Register model in MLflow Model Registry
            try:
                from src.config import MODEL_NAME
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                
                # Register model in Model Registry
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=MODEL_NAME
                )
                print(f"\n✓ Model registered in MLflow Model Registry")
                print(f"  Model Name: {MODEL_NAME}")
                print(f"  Version: {registered_model.version}")
                
                # Transition to Staging stage (can be promoted to Production later)
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=registered_model.version,
                    stage="Staging"
                )
                print(f"  Stage: Staging")
                
            except Exception as reg_error:
                print(f"\n⚠️  Warning: Could not register model in Model Registry: {reg_error}")
                print("  Model is still logged in MLflow, but not registered.")
            
            print(f"\n✓ MLflow run completed!")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
            
            return str(model_path)
            
    except Exception as e:
        print(f"Error during training or MLflow tracking: {str(e)}")
        # Only save model if it was successfully trained
        if 'model' in locals() and model is not None:
            try:
                model_dir = Path(MODELS_DIR)
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / 'stock_model.pkl'
                joblib.dump(model, model_path)
                print(f"Model saved locally despite error: {model_path}")
            except Exception as save_error:
                print(f"Failed to save model: {str(save_error)}")
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

