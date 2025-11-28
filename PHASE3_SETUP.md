# Phase 3: CI/CD Pipeline Setup Guide

This document provides instructions for setting up and using the Phase 3 CI/CD pipeline.

## Overview

Phase 3 implements:
- FastAPI REST API for model serving
- Docker containerization for the API
- GitHub Actions CI/CD workflows
- Automated testing and deployment

## Files Created

### 1. FastAPI Application
- **File**: `src/api.py`
- **Purpose**: REST API for serving stock price predictions
- **Endpoints**:
  - `GET /` - API information
  - `GET /health` - Health check
  - `POST /predict` - Make predictions
  - `GET /metrics` - Prometheus metrics

### 2. Docker Configuration
- **File**: `docker/api/Dockerfile`
- **Purpose**: Container image for the API service
- **Usage**: `docker build -f docker/api/Dockerfile -t mlops-api .`

### 3. GitHub Actions Workflows

#### CI - Feature to Dev (`ci-feature-to-dev.yml`)
- **Triggers**: PRs to `dev` branch
- **Actions**:
  - Code linting (flake8)
  - Unit tests (pytest)
  - Coverage reports

#### CI - Dev to Test (`ci-dev-to-test.yml`)
- **Triggers**: PRs to `test` branch
- **Actions**:
  - Train model
  - Compare with baseline using CML
  - Block merge if model performance degrades

#### CD - Test to Main (`cd-test-to-main.yml`)
- **Triggers**: PRs to `main` branch
- **Actions**:
  - Build Docker image
  - Push to Docker Hub
  - Verify deployment

## Setup Instructions

### 1. GitHub Secrets Configuration

Add the following secrets to your GitHub repository:

1. Go to: Settings → Secrets and variables → Actions → New repository secret

**Required Secrets:**
- `AWS_ACCESS_KEY_ID` - AWS S3 access key
- `AWS_SECRET_ACCESS_KEY` - AWS S3 secret key
- `MLFLOW_TRACKING_URI` - MLflow tracking URI (Dagshub)
- `DAGSHUB_USERNAME` - Dagshub username
- `DAGSHUB_TOKEN` - Dagshub token
- `ALPHA_VANTAGE_KEY` - Alpha Vantage API key
- `DOCKER_HUB_USERNAME` - Docker Hub username
- `DOCKER_HUB_TOKEN` - Docker Hub access token

### 2. Docker Hub Setup

1. Create account at https://hub.docker.com
2. Generate access token: Account Settings → Security → New Access Token
3. Add `DOCKER_HUB_USERNAME` and `DOCKER_HUB_TOKEN` to GitHub secrets

### 3. Local Testing

#### Test FastAPI Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python src/api.py

# Or with uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

#### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [150.0, 152.0, 149.0, 151.0, 1000000.0, 0.5, 0.3, 0.2]}'

# Get metrics
curl http://localhost:8000/metrics
```

#### Build and Run Docker Image
```bash
# Build image
docker build -f docker/api/Dockerfile -t mlops-stock-prediction-api .

# Run container
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
  mlops-stock-prediction-api

# Or use docker-compose
docker-compose -f docker-compose.api.yml up
```

### 4. Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov flake8

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run linting
flake8 src/ tests/
```

## Branch Protection Rules & PR Approvals

**IMPORTANT**: Configure branch protection rules in GitHub to enforce PR approvals (Required by Phase III Step 5.3):

1. Go to: Settings → Branches → Add rule

### For `test` branch:
- **Require pull request reviews before merging** (1 approval minimum)
- Require status checks to pass:
  - `train-and-compare`
- Include administrators

### For `main` branch:
- **Require pull request reviews before merging** (1 approval minimum)
- Require status checks to pass:
  - `build-and-deploy`
- Include administrators

**Note**: This enforces the mandatory PR approval requirement (Step 5.3) for merging into `test` and `main` branches.

## Workflow Usage

### Feature → Dev
1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and commit
3. Create PR to `dev` branch
4. CI workflow runs automatically:
   - Linting checks
   - Unit tests
   - Coverage reports

### Dev → Test
1. Create PR from `dev` to `test`
2. CI workflow runs:
   - Trains new model
   - Compares with baseline
   - Posts CML report as PR comment
3. If model performance is worse, merge is blocked
4. Requires approval before merging

### Test → Main
1. Create PR from `test` to `main`
2. CD workflow runs:
   - Builds Docker image
   - Pushes to Docker Hub
   - Verifies deployment
3. Requires approval before merging
4. After merge, creates deployment tag

## Monitoring

The API exposes Prometheus metrics at `/metrics`:
- `api_requests_total` - Total API requests
- `api_inference_latency_seconds` - Inference latency
- `data_drift_detected_total` - Data drift detections

## Troubleshooting

### API fails to start
- Check if model file exists: `ls models/stock_model.pkl`
- Verify model was trained: Run `python src/train.py`
- Check environment variables are set

### Docker build fails
- Verify all dependencies in `requirements.txt`
- Check Dockerfile syntax
- Ensure Python 3.8 compatibility

### GitHub Actions fails
- Check secrets are configured correctly
- Verify branch names match workflow triggers
- Check workflow logs for specific errors

### CML not working
- Verify `DAGSHUB_TOKEN` is set
- Check CML installation in workflow
- Ensure model training completed successfully

## Next Steps

After Phase 3 completion:
- Phase 4: Set up Prometheus and Grafana for monitoring
- Configure alerting rules
- Set up production deployment infrastructure

