"""
Airflow DAG for Stock Prediction Pipeline
Orchestrates the entire ETL and model training workflow
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to Python path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our custom modules
from src.data_extraction import fetch_stock_data
from src.data_transformation import transform_data

# Default arguments for all tasks
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,  # Don't run past dates
}

# Create DAG
dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='ETL and ML pipeline for stock price prediction using Alpha Vantage API',
    schedule_interval='@daily',  # Run once per day at midnight
    catchup=False,  # Don't backfill
    tags=['mlops', 'stock-prediction', 'etl'],
)


# ==================== Task 1: Extract Data ====================
def extract_task():
    """
    Extract stock data from Alpha Vantage API
    Returns the file path for use in next tasks
    """
    print("Starting data extraction...")
    try:
        file_path = fetch_stock_data()
        print(f"Extraction completed. File: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error in extraction: {str(e)}")
        raise


extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_task,
    dag=dag,
)


# ==================== Task 2: Data Quality Check ====================
def quality_check_task(**context):
    """
    Check data quality
    Uses XCom to get file path from extract task
    """
    # Get file path from previous task
    ti = context['ti']
    input_path = ti.xcom_pull(task_ids='extract_data')
    
    if not input_path:
        raise ValueError("No file path received from extract_data task")
    
    print(f"Checking data quality for: {input_path}")
    
    # Import here to avoid issues if module not available
    from src.data_quality_check import check_data_quality
    
    # Run quality check
    passed = check_data_quality(input_path)
    
    if not passed:
        raise ValueError(f"Data quality check failed for {input_path}")
    
    print("Data quality check passed!")
    return input_path


quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=quality_check_task,
    provide_context=True,
    dag=dag,
)


# ==================== Task 3: Transform Data ====================
def transform_task(**context):
    """
    Transform data and create features
    Uses XCom to get file path from quality check task
    """
    # Get file path from previous task
    ti = context['ti']
    input_path = ti.xcom_pull(task_ids='data_quality_check')
    
    if not input_path:
        raise ValueError("No file path received from data_quality_check task")
    
    print(f"Transforming data from: {input_path}")
    
    # Transform data
    output_path, report_path = transform_data(input_path, generate_report=True)
    
    print(f"Transformation completed. Output: {output_path}")
    if report_path:
        print(f"Report generated: {report_path}")
    
    return output_path


transform = PythonOperator(
    task_id='transform_data',
    python_callable=transform_task,
    provide_context=True,
    dag=dag,
)


# ==================== Task 4: Version Data with DVC ====================
def dvc_version_task(**context):
    """
    Version the processed data with DVC
    Uses XCom to get file path from transform task
    """
    # Get file path from previous task
    ti = context['ti']
    file_path = ti.xcom_pull(task_ids='transform_data')
    
    if not file_path:
        raise ValueError("No file path received from transform_data task")
    
    print(f"Versioning data with DVC: {file_path}")
    
    # Change to project root directory
    os.chdir(project_root)
    
    # DVC commands
    import subprocess
    
    # Add file to DVC tracking
    result = subprocess.run(
        ['dvc', 'add', file_path],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    
    if result.returncode != 0:
        print(f"DVC add warning: {result.stderr}")
        # Continue anyway - might be already tracked
    
    # Git add DVC files
    dvc_file = file_path + '.dvc'
    if os.path.exists(dvc_file):
        subprocess.run(
            ['git', 'add', dvc_file, '.dvc/config'],
            cwd=str(project_root),
            capture_output=True
        )
        
        # Commit (optional - might fail if nothing to commit)
        commit_msg = f"Update data version {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=str(project_root),
            capture_output=True
        )
    
    # Push to DVC remote
    result = subprocess.run(
        ['dvc', 'push'],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    
    if result.returncode != 0:
        print(f"DVC push warning: {result.stderr}")
        # Don't fail the task - might be network issue
    
    print("DVC versioning completed")
    return file_path


dvc_version = PythonOperator(
    task_id='version_with_dvc',
    python_callable=dvc_version_task,
    provide_context=True,
    dag=dag,
)


# ==================== Task 5: Train Model ====================
def train_model_task(**context):
    """
    Train the ML model
    Uses XCom to get processed file path from transform task
    """
    # Get file path from previous task
    ti = context['ti']
    data_path = ti.xcom_pull(task_ids='transform_data')
    
    if not data_path:
        raise ValueError("No data path received from transform_data task")
    
    print(f"Training model with data from: {data_path}")
    
    # Change to project root
    os.chdir(project_root)
    
    # Run training script
    import subprocess
    
    result = subprocess.run(
        [sys.executable, 'src/train.py', data_path],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Training error: {result.stderr}")
        raise Exception(f"Model training failed: {result.stderr}")
    
    print("Model training completed successfully!")
    print(result.stdout)
    
    return data_path


train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)


# ==================== Define Task Dependencies ====================
# Set up the pipeline flow:
# extract → quality_check → transform → dvc_version → train_model

extract >> quality_check >> transform >> dvc_version >> train_model

