# LiveInsight+ System Launcher WITH Agent
# Starts all services including autonomous LIME-powered agent
# PowerShell script for Windows

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "🤖 LiveInsight+ WITH Autonomous Agent - System Launcher" -ForegroundColor Green
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

# Check if Kafka is needed
Write-Host "Checking for Kafka (optional for streaming)..." -ForegroundColor Yellow
Write-Host "If you don't have Kafka running, the processor will wait for data." -ForegroundColor Gray
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
New-Item -ItemType Directory -Force -Path "output/xai/shap" | Out-Null
New-Item -ItemType Directory -Force -Path "output/xai/lime" | Out-Null
New-Item -ItemType Directory -Force -Path "output/agent_lime_decisions" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Write-Host "✅ Directories created" -ForegroundColor Green
Write-Host ""

# Launch services
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "🚀 Launching Services..." -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting services in separate windows..." -ForegroundColor Yellow
Write-Host "Each service will open in its own terminal window." -ForegroundColor Gray
Write-Host ""

# 1. Kafka Producer
Write-Host "[1/5] Starting Kafka Producer..." -ForegroundColor Cyan
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
Write-Host "[2/5] Starting Stream Processor..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '⚙️  Stream Processor - Processing Kafka events' -ForegroundColor Green;
Write-Host 'Press Ctrl+C to stop' -ForegroundColor Yellow;
Write-Host '';
python processor_consumer.py --checkpoint 3
"@
Start-Sleep -Seconds 2
Write-Host "✅ Stream Processor started" -ForegroundColor Green
Write-Host ""

# 3. Enhanced ML Service with LIME
Write-Host "[3/5] Starting Enhanced ML + XAI Service (Random Forest + SHAP + LIME)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '🧠 ML + XAI Service - Random Forest with SHAP and LIME' -ForegroundColor Green;
Write-Host 'API: http://localhost:8000' -ForegroundColor Yellow;
Write-Host 'Docs: http://localhost:8000/docs' -ForegroundColor Yellow;
Write-Host '';
python ml_service_enhanced.py
"@
Start-Sleep -Seconds 3
Write-Host "✅ ML Service started on http://localhost:8000" -ForegroundColor Green
Write-Host ""

# 4. LIME-Powered Autonomous Agent
Write-Host "[4/5] Starting LIME-Powered Autonomous Agent..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '🤖 Autonomous Agent - LIME-Powered Decision Making' -ForegroundColor Green;
Write-Host 'This agent uses LIME explanations for transparent decisions' -ForegroundColor Yellow;
Write-Host '';
python agent_with_lime.py --interval 30
"@
Start-Sleep -Seconds 2
Write-Host "✅ Autonomous Agent started (30s cycle)" -ForegroundColor Green
Write-Host ""

# 5. Dashboard WITH Agent
Write-Host "[5/5] Starting Dashboard WITH Agent..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host '📊 Dashboard WITH Agent - http://localhost:8501' -ForegroundColor Green;
Write-Host 'This dashboard shows autonomous agent decisions with LIME explanations' -ForegroundColor Yellow;
Write-Host '';
streamlit run dashboard_with_agent.py
"@
Write-Host "✅ Dashboard started on http://localhost:8501" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "✅ ALL SERVICES STARTED - WITH AGENT MODE" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "🔗 Access Points:" -ForegroundColor Yellow
Write-Host "   Dashboard (WITH Agent): http://localhost:8501" -ForegroundColor White
Write-Host "   ML API: http://localhost:8000" -ForegroundColor White
Write-Host "   ML Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

Write-Host "📁 Output Files:" -ForegroundColor Yellow
Write-Host "   Agent Actions: output/agent_actions_with_lime.csv" -ForegroundColor White
Write-Host "   LIME Decisions: output/agent_lime_decisions/" -ForegroundColor White
Write-Host "   Predictions: output/predictions.csv" -ForegroundColor White
Write-Host "   Metrics: output/agent_performance_metrics.csv" -ForegroundColor White
Write-Host ""

Write-Host "🤖 Agent Features:" -ForegroundColor Yellow
Write-Host "   ✅ Autonomous monitoring (24/7)" -ForegroundColor Green
Write-Host "   ✅ LIME explanations for every decision" -ForegroundColor Green
Write-Host "   ✅ Human-readable justifications" -ForegroundColor Green
Write-Host "   ✅ Auto-approval for high-confidence decisions" -ForegroundColor Green
Write-Host "   ✅ Complete audit trail" -ForegroundColor Green
Write-Host ""

Write-Host "⚠️  To stop all services:" -ForegroundColor Yellow
Write-Host "   Close all PowerShell windows or press Ctrl+C in each" -ForegroundColor Gray
Write-Host ""

Write-Host "💡 Tip: Compare with WITHOUT agent version!" -ForegroundColor Cyan
Write-Host "   Run: .\start_system_without_agent.ps1" -ForegroundColor White
Write-Host "   Dashboard: http://localhost:8502" -ForegroundColor White
Write-Host ""

Write-Host "Press any key to exit this launcher..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
