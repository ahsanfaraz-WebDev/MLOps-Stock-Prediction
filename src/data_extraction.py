"""
Data Extraction Script for Stock Prediction Pipeline
Fetches intraday stock data from Alpha Vantage API
"""
import requests
import pandas as pd
from datetime import datetime
import os
import sys
from pathlib import Path

# Import configuration
from src.config import ALPHA_VANTAGE_KEY, STOCK_SYMBOL, DATA_RAW_DIR


def fetch_stock_data(symbol=None, api_key=None):
    """
    Fetch intraday stock data from Alpha Vantage API
    
    Args:
        symbol: Stock symbol (default: from config)
        api_key: Alpha Vantage API key (default: from config)
    
    Returns:
        str: Path to saved CSV file
    
    Raises:
        Exception: If API call fails or returns error
    """
    # Use provided values or fall back to config
    symbol = symbol or STOCK_SYMBOL
    api_key = api_key or ALPHA_VANTAGE_KEY
    
    if not api_key:
        raise ValueError("Alpha Vantage API key is required. Set ALPHA_VANTAGE_KEY in .env file")
    
    if not symbol:
        raise ValueError("Stock symbol is required. Set STOCK_SYMBOL in .env file")
    
    # API endpoint
    url = 'https://www.alphavantage.co/query'
    
    # API parameters
    # Note: 'compact' returns last 100 data points (free tier)
    # 'full' requires premium API key and may hit rate limits
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '60min',  # 1 hour intervals
        'apikey': api_key,
        'outputsize': 'compact',  # Use 'compact' for free tier (last 100 data points)
        'datatype': 'json'
    }
    
    print(f"Fetching data for {symbol} from Alpha Vantage API...")
    print(f"API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
    
    try:
        # Make API request
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise Exception(f"API Error: {data['Error Message']}")
        
        if 'Note' in data:
            raise Exception(f"API Rate Limit: {data['Note']}")
        
        # Check for Information message (usually rate limit or API key issue)
        if 'Information' in data:
            info_msg = data['Information']
            if 'API call frequency' in info_msg or 'rate' in info_msg.lower():
                raise Exception(f"API Rate Limit: {info_msg}")
            else:
                raise Exception(f"API Information: {info_msg}. This usually means rate limit exceeded or API key issue.")
        
        # Extract time series data
        time_series_key = 'Time Series (60min)'
        if time_series_key not in data:
            # Try alternative key names
            possible_keys = [k for k in data.keys() if 'Time Series' in k]
            if possible_keys:
                time_series_key = possible_keys[0]
            else:
                # Log the full response for debugging
                print(f"DEBUG: Full API response: {data}")
                raise Exception(f"No time series data found. Response keys: {list(data.keys())}. Full response: {data}")
        
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            raise Exception("No data returned from API. Check API key and symbol.")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns (Alpha Vantage uses numbered keys)
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Ensure we have all required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise Exception(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date (oldest first)
        df = df.sort_index()
        
        # Create output directory if it doesn't exist
        output_dir = Path(DATA_RAW_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        fetch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        output_filename = f'stock_data_{symbol}_{fetch_timestamp}.csv'
        output_path = output_dir / output_filename
        
        df.to_csv(output_path)
        
        print(f"✓ Data saved successfully!")
        print(f"  File: {output_path}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {', '.join(df.columns)}")
        
        return str(output_path)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error while fetching data: {str(e)}")
    except Exception as e:
        raise Exception(f"Error fetching stock data: {str(e)}")


if __name__ == "__main__":
    """
    Allow script to be run directly from command line
    """
    try:
        # Check for command line arguments
        symbol = sys.argv[1] if len(sys.argv) > 1 else None
        api_key = sys.argv[2] if len(sys.argv) > 2 else None
        
        file_path = fetch_stock_data(symbol=symbol, api_key=api_key)
        print(f"\n✓ Extraction completed successfully!")
        print(f"Output file: {file_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)



