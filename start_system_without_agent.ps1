# LiveInsight+ System Launcher WITHOUT Agent
# Starts services for manual inventory management (no autonomous agent)
# PowerShell script for Windows

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "📋 LiveInsight+ WITHOUT Agent - Manual Mode Launcher" -ForegroundColor Yellow
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found! Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}
Write-Host "✅ $pythonVersion" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Some dependencies may have failed to install" -ForegroundColor Yellow
}
Write-Host "✅ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Create output directories
Write-Host "Creating output directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "output/xai" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Write-Host "✅ Directories created" -ForegroundColor Green
Write-Host ""

# Launch services
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "🚀 Launching Services (Manual Mode - NO AGENT)..." -ForegroundColor Yellow
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting services in separate windows..." -ForegroundColor Yellow
Write-Host "Each service will open in its own terminal window." -ForegroundColor Gray
Write-Host ""

# 1. Kafka Producer
Write-Host "[1/4] Starting Kafka Producer..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '🔄 Kafka Producer - Streaming retail_data_bangalore.csv' -ForegroundColor Green;
Write-Host 'Press Ctrl+C to stop' -ForegroundColor Yellow;
Write-Host '';
python stream_server_kafka.py --delay 0.05 --loop
"@
Start-Sleep -Seconds 2
Write-Host "✅ Kafka Producer started" -ForegroundColor Green
Write-Host ""

# 2. Stream Processor
Write-Host "[2/4] Starting Stream Processor..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '⚙️  Stream Processor - Processing Kafka events' -ForegroundColor Green;
Write-Host 'Press Ctrl+C to stop' -ForegroundColor Yellow;
Write-Host '';
python processor_consumer.py --checkpoint 3
"@
Start-Sleep -Seconds 2
Write-Host "✅ Stream Processor started" -ForegroundColor Green
Write-Host ""

# 3. Enhanced ML Service (predictions available but no agent)
Write-Host "[3/4] Starting ML + XAI Service..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '🧠 ML + XAI Service - Predictions for Manual Review' -ForegroundColor Green;
Write-Host 'API: http://localhost:8000' -ForegroundColor Yellow;
Write-Host 'Docs: http://localhost:8000/docs' -ForegroundColor Yellow;
Write-Host '';
python ml_service_enhanced.py
"@
Start-Sleep -Seconds 3
Write-Host "✅ ML Service started on http://localhost:8000" -ForegroundColor Green
Write-Host ""

Write-Host "⚠️  SKIPPING Agent - Manual Mode" -ForegroundColor Yellow
Write-Host "   No autonomous agent will run" -ForegroundColor Gray
Write-Host "   All decisions require manual human review" -ForegroundColor Gray
Write-Host ""

# 4. Dashboard WITHOUT Agent (on different port to allow side-by-side)
Write-Host "[4/4] Starting Dashboard WITHOUT Agent..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '📊 Dashboard WITHOUT Agent - http://localhost:8502' -ForegroundColor Yellow;
Write-Host 'Manual inventory management mode' -ForegroundColor Yellow;
Write-Host '';
streamlit run dashboard_without_agent.py --server.port 8502
"@
Write-Host "✅ Dashboard started on http://localhost:8502" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "✅ ALL SERVICES STARTED - WITHOUT AGENT (MANUAL MODE)" -ForegroundColor Yellow
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "🔗 Access Points:" -ForegroundColor Yellow
Write-Host "   Dashboard (WITHOUT Agent): http://localhost:8502" -ForegroundColor White
Write-Host "   ML API: http://localhost:8000" -ForegroundColor White
Write-Host "   ML Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

Write-Host "📁 Output Files:" -ForegroundColor Yellow
Write-Host "   Predictions: output/predictions.csv" -ForegroundColor White
Write-Host "   Inventory: output/product_inventory_usage.csv" -ForegroundColor White
Write-Host "   Sales Data: output/branch_sales.csv, output/product_sales.csv" -ForegroundColor White
Write-Host ""

Write-Host "📋 Manual Mode Features:" -ForegroundColor Yellow
Write-Host "   ⚠️  No autonomous agent" -ForegroundColor Red
Write-Host "   ⚠️  Manual review required for all products" -ForegroundColor Red
Write-Host "   ⚠️  Human must check dashboards regularly" -ForegroundColor Red
Write-Host "   ⚠️  Manual calculation of reorder quantities" -ForegroundColor Red
Write-Host "   ⚠️  Manager approval required for every order" -ForegroundColor Red
Write-Host ""

Write-Host "⚠️  To stop all services:" -ForegroundColor Yellow
Write-Host "   Close all PowerShell windows or press Ctrl+C in each" -ForegroundColor Gray
Write-Host ""

Write-Host "💡 Compare with agent version!" -ForegroundColor Cyan
Write-Host "   Run in another terminal: .\start_system_with_agent.ps1" -ForegroundColor White
Write-Host "   Dashboard WITH Agent: http://localhost:8501" -ForegroundColor White
Write-Host "   Dashboard WITHOUT Agent: http://localhost:8502" -ForegroundColor White
Write-Host ""
Write-Host "   Open BOTH dashboards side-by-side to see the difference!" -ForegroundColor Green
Write-Host ""

Write-Host "Press any key to exit this launcher..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
