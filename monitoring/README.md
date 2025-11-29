# Monitoring and Observability Setup

This directory contains the configuration files for Prometheus and Grafana monitoring setup.

## Overview

Phase IV implements comprehensive monitoring and observability for the MLOps Stock Prediction system using:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and alerting dashboards

## Architecture

```
FastAPI API (Port 8000)
    ↓ (exposes /metrics endpoint)
Prometheus (Port 9090)
    ↓ (scrapes metrics every 15s)
Grafana (Port 3000)
    ↓ (visualizes metrics)
Dashboard & Alerts
```

## Metrics Collected

### Service Metrics
- **API Request Rate**: Total requests per second by method, endpoint, and status
- **Inference Latency**: p50, p95, p99 percentiles of prediction latency
- **Total Requests**: Cumulative request count

### Model/Data Drift Metrics
- **Data Drift Ratio**: Ratio of requests with out-of-distribution features
- **Data Drift Count**: Total number of drift detections per second
- **Prediction Requests**: Total prediction requests per second

## Setup Instructions

### 1. Start Services

```bash
# Start API, Prometheus, and Grafana
docker-compose -f docker-compose.api.yml up -d

# Check service status
docker-compose -f docker-compose.api.yml ps
```

### 2. Access Services

- **FastAPI API**: http://localhost:8000
  - Health: http://localhost:8000/health
  - Metrics: http://localhost:8000/metrics
  - Docs: http://localhost:8000/docs

- **Prometheus**: http://localhost:9090
  - Targets: http://localhost:9090/targets
  - Query: http://localhost:9090/graph

- **Grafana**: http://localhost:3000
  - Default credentials: `admin` / `admin` (change on first login)
  - Dashboard: Automatically loaded from provisioning

### 3. Verify Metrics Collection

1. **Check Prometheus Targets**:
   - Navigate to http://localhost:9090/targets
   - Verify `fastapi-api` target is UP

2. **Query Metrics in Prometheus**:
   - Go to http://localhost:9090/graph
   - Try queries:
     - `rate(api_requests_total[5m])`
     - `histogram_quantile(0.95, rate(api_inference_latency_seconds_bucket[5m]))`
     - `rate(data_drift_detected_total[5m]) / rate(prediction_requests_total[5m])`

3. **View Grafana Dashboard**:
   - Login to Grafana
   - Navigate to Dashboards → MLOps Stock Prediction - Service & Model Health
   - Dashboard should show real-time metrics

### 4. Generate Test Traffic

```bash
# Make some prediction requests to generate metrics
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [150.0, 152.0, 149.0, 151.0, 1000000.0, 0.5, 0.3, 0.2]}'

# Make multiple requests to see metrics change
for i in {1..10}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [150.0, 152.0, 149.0, 151.0, 1000000.0, 0.5, 0.3, 0.2]}'
  sleep 1
done
```

## Alerting Configuration

Grafana alerts are configured for:

1. **High Inference Latency** (> 500ms)
   - Condition: p95 latency > 0.5s for 2 minutes
   - Severity: Warning

2. **Data Drift Spike** (> 10%)
   - Condition: Drift ratio > 10% for 5 minutes
   - Severity: Critical

3. **High Error Rate** (> 5%)
   - Condition: 5xx errors > 5% for 2 minutes
   - Severity: Warning

### Configuring Alert Notifications

To send alerts to Slack or other channels:

1. Go to Grafana → Alerting → Notification channels
2. Add new channel (Slack, Email, etc.)
3. Configure channel settings
4. Update alert rules to use the notification channel

## File Structure

```
monitoring/
├── prometheus.yml                    # Prometheus scrape configuration
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml       # Prometheus datasource config
│       ├── dashboards/
│       │   ├── dashboard.yml        # Dashboard provisioning config
│       │   └── mlops-dashboard.json # Dashboard definition
│       └── alerting/
│           └── alert-rules.yml      # Alert rules configuration
└── README.md                         # This file
```

## Troubleshooting

### Prometheus not scraping API

1. Check API is running: `docker-compose -f docker-compose.api.yml ps api`
2. Check API metrics endpoint: `curl http://localhost:8000/metrics`
3. Check Prometheus targets: http://localhost:9090/targets
4. Check Prometheus logs: `docker-compose -f docker-compose.api.yml logs prometheus`

### Grafana not showing data

1. Verify Prometheus datasource is configured correctly
2. Check datasource connection: Grafana → Configuration → Data Sources → Prometheus → Test
3. Verify time range in dashboard (should be "Last 1 hour" or similar)
4. Check Grafana logs: `docker-compose -f docker-compose.api.yml logs grafana`

### No metrics appearing

1. Generate some API traffic (see "Generate Test Traffic" above)
2. Wait 15-30 seconds for Prometheus to scrape
3. Check Prometheus query: `api_requests_total`
4. Verify API is exposing metrics: `curl http://localhost:8000/metrics`

## Customization

### Adding New Metrics

1. Add metric to `src/api.py`:
```python
from prometheus_client import Gauge, Counter, Histogram

NEW_METRIC = Counter('new_metric_total', 'Description')
```

2. Update Prometheus scrape config if needed
3. Add panel to Grafana dashboard JSON

### Modifying Alert Thresholds

Edit `monitoring/grafana/provisioning/alerting/alert-rules.yml`:
```yaml
- alert: HighInferenceLatency
  expr: histogram_quantile(0.95, rate(api_inference_latency_seconds_bucket[5m])) > 0.5
  # Change 0.5 to your desired threshold
```

## Cleanup

```bash
# Stop all services
docker-compose -f docker-compose.api.yml down

# Remove volumes (deletes all metrics and Grafana data)
docker-compose -f docker-compose.api.yml down -v
```


