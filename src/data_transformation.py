"""
i228791
Data Transformation Script
Performs feature engineering for time-series stock prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Import configuration
from src.config import DATA_PROCESSED_DIR

# Try to import ydata-profiling (replacement for pandas-profiling)
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    try:
        from pandas_profiling import ProfileReport
        PROFILING_AVAILABLE = True
    except ImportError:
        PROFILING_AVAILABLE = False
        print("Warning: ydata-profiling not available. Skipping profile report generation.")


def create_features(df):
    """
    Create time-series features for stock prediction
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
    
    Returns:
        DataFrame: DataFrame with added features
    """
    print("Creating features...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    print(f"  Starting with {len(df)} rows")
    
    # ==================== 1. Lag Features ====================
    # Previous values (1h, 2h, 3h, 6h, 12h, 24h ago)
    print("  Creating lag features...")
    lag_periods = [1, 2, 3, 6, 12, 24]
    
    for lag in lag_periods:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # ==================== 2. Rolling Statistics ====================
    # Rolling windows: 3h, 6h, 12h, 24h
    print("  Creating rolling statistics...")
    rolling_windows = [3, 6, 12, 24]
    
    for window in rolling_windows:
        # Rolling mean
        df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean()
        
        # Rolling standard deviation
        df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window, min_periods=1).std()
        
        # Rolling min/max
        df[f'close_rolling_min_{window}'] = df['close'].rolling(window=window, min_periods=1).min()
        df[f'close_rolling_max_{window}'] = df['close'].rolling(window=window, min_periods=1).max()
    
    # ==================== 3. Price Changes ====================
    print("  Creating price change features...")
    
    # Intraday price change
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
    
    # Price change from previous close
    df['close_change'] = df['close'].diff()
    df['close_change_pct'] = (df['close'].pct_change()) * 100
    
    # ==================== 4. High-Low Range ====================
    print("  Creating high-low range features...")
    
    df['high_low_range'] = df['high'] - df['low']
    df['high_low_range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
    
    # ==================== 5. Time-based Features ====================
    print("  Creating time-based features...")
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # ==================== 6. Technical Indicators ====================
    print("  Creating technical indicators...")
    
    # Simple Moving Average (SMA)
    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    # Price relative to SMA
    df['close_to_sma5'] = df['close'] / df['sma_5']
    df['close_to_sma10'] = df['close'] / df['sma_10']
    
    # Volume features
    df['volume_change'] = df['volume'].diff()
    df['volume_change_pct'] = df['volume'].pct_change() * 100
    
    # ==================== 7. Target Variable ====================
    # Next hour's closing price (what we want to predict)
    print("  Creating target variable...")
    df['target'] = df['close'].shift(-1)
    
    # ==================== 8. Additional Features ====================
    # Volatility (using rolling std)
    df['volatility'] = df['close'].rolling(window=12, min_periods=1).std()
    
    # Price position within day's range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    print(f"  Created {len(df.columns)} total features")
    
    # Drop rows with NaN (from lag/rolling operations)
    # Keep rows where we have enough data for features
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    print(f"  Dropped {dropped_rows} rows with NaN (from lag operations)")
    print(f"  Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def transform_data(input_path, generate_report=True):
    """
    Main transformation pipeline
    
    Args:
        input_path: Path to raw CSV file
        generate_report: Whether to generate data profiling report
    
    Returns:
        tuple: (output_path, report_path) - paths to processed CSV and HTML report
    """
    print(f"\n{'='*60}")
    print(f"Data Transformation Pipeline")
    print(f"{'='*60}")
    print(f"Input file: {input_path}")
    
    # Load raw data
    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} rows from {input_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
    # Create features
    df_processed = create_features(df)
    
    # Create output directory
    output_dir = Path(DATA_PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save processed data
    output_filename = f'stock_data_processed_{timestamp}.csv'
    output_path = output_dir / output_filename
    df_processed.to_csv(output_path)
    
    print(f"\n✓ Processed data saved to: {output_path}")
    
    # Generate data quality report
    report_path = None
    if generate_report and PROFILING_AVAILABLE:
        try:
            print(f"\nGenerating data profile report...")
            report_filename = f'data_profile_report_{timestamp}.html'
            report_path = output_dir / report_filename
            
            # Create minimal profile (faster)
            profile = ProfileReport(
                df_processed,
                title="Stock Data Profile",
                minimal=True,
                progress_bar=False
            )
            profile.to_file(report_path)
            print(f"✓ Data profile report saved to: {report_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not generate profile report: {str(e)}")
            report_path = None
    elif generate_report and not PROFILING_AVAILABLE:
        print(f"⚠ Skipping profile report (ydata-profiling not installed)")
    
    print(f"\n{'='*60}")
    print(f"Transformation completed successfully!")
    print(f"{'='*60}\n")
    
    return str(output_path), str(report_path) if report_path else None


if __name__ == "__main__":
    """
    Allow script to be run directly from command line
    Usage: python data_transformation.py <input_path> [--no-report]
    """
    if len(sys.argv) < 2:
        print("Usage: python data_transformation.py <input_path> [--no-report]")
        print("Example: python data_transformation.py data/raw/stock_data_AAPL_20241125_120000.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    generate_report = '--no-report' not in sys.argv
    
    try:
        output_path, report_path = transform_data(input_path, generate_report=generate_report)
        print(f"Output file: {output_path}")
        if report_path:
            print(f"Report file: {report_path}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)



