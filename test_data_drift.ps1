# Test Data Drift Detection
# This script generates requests that will trigger data drift detection

Write-Host "=== Testing Data Drift Detection ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Sending 20 requests with DRIFT (negative values)..." -ForegroundColor Yellow

$driftCount = 0
for ($i=1; $i -le 20; $i++) {
    try {
        # Send request with negative value (triggers drift detection)
        $body = @{features=@(-10.0, 155.0, 152.0, 155.0, 155.0, 150.0)} | ConvertTo-Json
        Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body | Out-Null
        $driftCount++
        Write-Host "  Drift request $i sent" -ForegroundColor Gray
        Start-Sleep -Milliseconds 300
    } catch {
        Write-Host "  Request $i failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[OK] Sent $driftCount drift requests" -ForegroundColor Green
Write-Host ""
Write-Host "Waiting 20 seconds for Prometheus to scrape..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

Write-Host ""
Write-Host "=== Checking Metrics ===" -ForegroundColor Cyan

# Check drift metrics
try {
    $drift = (Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=data_drift_detected_total" -UseBasicParsing).Content | ConvertFrom-Json
    $total = (Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=prediction_requests_total" -UseBasicParsing).Content | ConvertFrom-Json
    
    $driftValue = [int]$drift.data.result[0].value[1]
    $totalValue = [int]$total.data.result[0].value[1]
    $ratio = if ($totalValue -gt 0) { [math]::Round(($driftValue / $totalValue) * 100, 2) } else { 0 }
    
    Write-Host ""
    Write-Host "Data Drift Metrics:" -ForegroundColor Yellow
    Write-Host "  Total Predictions: $totalValue" -ForegroundColor White
    
    $driftColor = if ($driftValue -gt 0) { "Yellow" } else { "Gray" }
    Write-Host "  Drift Detections: $driftValue" -ForegroundColor $driftColor
    
    $ratioColor = if ($ratio -gt 10) { "Red" } elseif ($ratio -gt 0) { "Yellow" } else { "Green" }
    Write-Host "  Drift Ratio: $ratio%" -ForegroundColor $ratioColor
    
    if ($driftValue -gt 0) {
        Write-Host ""
        Write-Host "[OK] Data drift detection is working!" -ForegroundColor Green
        Write-Host "  Check Grafana dashboard 'Data Drift Ratio' panel" -ForegroundColor Cyan
        Write-Host "  Check Prometheus: http://localhost:9090" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "[WARNING] No drift detected yet. Try running the script again." -ForegroundColor Yellow
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] Error checking metrics: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Open Grafana: http://localhost:3000" -ForegroundColor White
Write-Host "2. Go to Alerting - Alert rules" -ForegroundColor White
Write-Host "3. Create alerts for High Latency and Data Drift" -ForegroundColor White
