"""
Unit tests for data quality check module
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_quality_check import check_data_quality

def test_check_data_quality_valid_data():
    """Test data quality check with valid data"""
    # Create valid test data
    data = {
        'timestamp': ['2025-01-01 10:00:00', '2025-01-01 11:00:00'],
        'open': [150.0, 151.0],
        'high': [152.0, 153.0],
        'low': [149.0, 150.0],
        'close': [151.0, 152.0],
        'volume': [1000000, 1100000]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        result = check_data_quality(temp_path)
        assert result == True
    finally:
        os.unlink(temp_path)

def test_check_data_quality_empty_dataframe():
    """Test data quality check with empty DataFrame"""
    df = pd.DataFrame()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        result = check_data_quality(temp_path)
        assert result == False
    finally:
        os.unlink(temp_path)

def test_check_data_quality_missing_columns():
    """Test data quality check with missing required columns"""
    data = {
        'timestamp': ['2025-01-01 10:00:00'],
        'open': [150.0]
        # Missing: high, low, close, volume
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        result = check_data_quality(temp_path)
        assert result == False
    finally:
        os.unlink(temp_path)

def test_check_data_quality_negative_prices():
    """Test data quality check with negative prices"""
    data = {
        'timestamp': ['2025-01-01 10:00:00'],
        'open': [-150.0],  # Negative price
        'high': [152.0],
        'low': [149.0],
        'close': [151.0],
        'volume': [1000000]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        result = check_data_quality(temp_path)
        assert result == False
    finally:
        os.unlink(temp_path)

def test_check_data_quality_invalid_high_low():
    """Test data quality check with high < low"""
    data = {
        'timestamp': ['2025-01-01 10:00:00'],
        'open': [150.0],
        'high': [149.0],  # High < Low (invalid)
        'low': [151.0],
        'close': [150.0],
        'volume': [1000000]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        result = check_data_quality(temp_path)
        assert result == False
    finally:
        os.unlink(temp_path)


