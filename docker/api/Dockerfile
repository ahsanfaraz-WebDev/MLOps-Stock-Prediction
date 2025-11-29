# Dockerfile for FastAPI Stock Prediction API
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=5 -r /tmp/requirements.txt

# Copy application code
COPY src/ /app/src/
COPY models/ /app/models/
# Note: .env file should be provided via environment variables in production

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

