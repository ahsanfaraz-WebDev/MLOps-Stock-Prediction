"""
Data Quality Check Script
Validates data quality before processing
"""
import pandas as pd
import sys
from pathlib import Path
from src.config import NULL_THRESHOLD


def check_data_quality(file_path, null_threshold=None):
    """
    Check data quality
    
    Args:
        file_path: Path to CSV file to check
        null_threshold: Maximum allowed percentage of null values (default: from config)
    
    Returns:
        bool: True if passes all checks, False otherwise
    """
    null_threshold = null_threshold or NULL_THRESHOLD
    
    print(f"\n{'='*60}")
    print(f"Data Quality Check: {file_path}")
    print(f"{'='*60}")
    
    # Check 1: File exists
    if not Path(file_path).exists():
        print(f"✗ FAIL: File does not exist: {file_path}")
        return False
    print(f"✓ File exists")
    
    # Check 2: Read file
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"✗ FAIL: Cannot read file: {str(e)}")
        return False
    print(f"✓ File readable")
    
    # Check 3: No empty dataframe
    if df.empty:
        print(f"✗ FAIL: DataFrame is empty")
        return False
    print(f"✓ DataFrame is not empty ({len(df)} rows)")
    
    # Check 4: Required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ FAIL: Missing required columns: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        return False
    print(f"✓ All required columns present: {', '.join(required_cols)}")
    
    # Check 5: Null value threshold
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    null_percentage = null_cells / total_cells if total_cells > 0 else 0
    
    print(f"\nNull Value Analysis:")
    print(f"  Total cells: {total_cells}")
    print(f"  Null cells: {null_cells}")
    print(f"  Null percentage: {null_percentage:.2%}")
    print(f"  Threshold: {null_threshold:.2%}")
    
    if null_percentage > null_threshold:
        print(f"✗ FAIL: Null percentage {null_percentage:.2%} exceeds threshold {null_threshold:.2%}")
        return False
    print(f"✓ Null percentage within threshold")
    
    # Check 6: Column-specific null checks
    print(f"\nColumn-wise Null Check:")
    for col in required_cols:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        status = "✓" if null_pct < (null_threshold * 100) else "✗"
        print(f"  {status} {col}: {null_count} nulls ({null_pct:.2f}%)")
        if null_pct >= (null_threshold * 100):
            print(f"    WARNING: High null percentage in {col}")
    
    # Check 7: No negative prices
    price_cols = ['open', 'high', 'low', 'close']
    negative_prices = (df[price_cols] < 0).any().any()
    if negative_prices:
        print(f"\n✗ FAIL: Negative prices detected")
        for col in price_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"  {col}: {neg_count} negative values")
        return False
    print(f"\n✓ No negative prices detected")
    
    # Check 8: Volume is non-negative
    negative_volume = (df['volume'] < 0).any()
    if negative_volume:
        print(f"✗ FAIL: Negative volume detected")
        neg_count = (df['volume'] < 0).sum()
        print(f"  volume: {neg_count} negative values")
        return False
    print(f"✓ No negative volume detected")
    
    # Check 9: Logical consistency (high >= low, high >= open, high >= close, etc.)
    print(f"\nLogical Consistency Checks:")
    
    # High should be >= Low
    high_low_invalid = (df['high'] < df['low']).sum()
    if high_low_invalid > 0:
        print(f"✗ FAIL: {high_low_invalid} rows where high < low")
        return False
    print(f"✓ High >= Low for all rows")
    
    # High should be >= Open and Close
    high_open_invalid = (df['high'] < df['open']).sum()
    high_close_invalid = (df['high'] < df['close']).sum()
    if high_open_invalid > 0 or high_close_invalid > 0:
        print(f"✗ FAIL: High price inconsistencies")
        if high_open_invalid > 0:
            print(f"  {high_open_invalid} rows where high < open")
        if high_close_invalid > 0:
            print(f"  {high_close_invalid} rows where high < close")
        return False
    print(f"✓ High >= Open and Close for all rows")
    
    # Low should be <= Open and Close
    low_open_invalid = (df['low'] > df['open']).sum()
    low_close_invalid = (df['low'] > df['close']).sum()
    if low_open_invalid > 0 or low_close_invalid > 0:
        print(f"✗ FAIL: Low price inconsistencies")
        if low_open_invalid > 0:
            print(f"  {low_open_invalid} rows where low > open")
        if low_close_invalid > 0:
            print(f"  {low_close_invalid} rows where low > close")
        return False
    print(f"✓ Low <= Open and Close for all rows")
    
    # Check 10: Data summary
    print(f"\n{'='*60}")
    print(f"Data Summary:")
    print(f"{'='*60}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nPrice Statistics:")
    print(df[price_cols].describe())
    print(f"\nVolume Statistics:")
    print(df[['volume']].describe())
    
    print(f"\n{'='*60}")
    print(f"✓ PASS: All quality checks passed!")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    """
    Allow script to be run directly from command line
    Usage: python data_quality_check.py <file_path>
    """
    if len(sys.argv) != 2:
        print("Usage: python data_quality_check.py <file_path>")
        print("Example: python data_quality_check.py data/raw/stock_data_AAPL_20241125_120000.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    passed = check_data_quality(file_path)
    
    # Exit with appropriate code (0 = success, 1 = failure)
    sys.exit(0 if passed else 1)



