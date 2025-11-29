# Use official Airflow image as base
FROM apache/airflow:2.7.0

# Switch to root to install system dependencies
USER root

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Set working directory
WORKDIR /opt/airflow

# Copy requirements file first (for better caching)
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies with increased timeout and retries
# Upgrade pip first, then install packagoes
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=5 -r /tmp/requirements.txt

# Set Python path
ENV PYTHONPATH=/opt/airflow

# Note: Project files are mounted as volume in docker-compose.yml
# No need to COPY them here

