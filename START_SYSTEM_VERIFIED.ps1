# LiveInsight+ System Launcher - Verified Version
# Starts complete system with multi-agent and both dashboards
# Updated with inventory tracking and negative stock prevention

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  LIVEINSIGHT+ COMPLETE SYSTEM LAUNCHER" -ForegroundColor Green
Write-Host "  Multi-Agent System with Inventory Tracking" -ForegroundColor Yellow
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$projectDir = "C:\Users\borut\Desktop\7 th sem el\retail-with-agent-and-explainable-ai"
$kafkaDir = "C:\kafka1"
$condaHook = "C:\Users\borut\anaconda3\shell\condabin\conda-hook.ps1"
$condaEnv = "hf-sentiment"

# Check prerequisites
Write-Host "[1/7] Checking Prerequisites..." -ForegroundColor Yellow

# Verify Kafka
if (Test-Path $kafkaDir) {
    Write-Host "  ✅ Kafka found" -ForegroundColor Green
    $useKafka = $true
} else {
    Write-Host "  ⚠️  Kafka not found - will run without streaming" -ForegroundColor Yellow
    $useKafka = $false
}

# Verify Python
if (Test-Path $condaHook) {
    Write-Host "  ✅ Conda environment ready" -ForegroundColor Green
} else {
    Write-Host "  ❌ Conda not found at expected path" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create output directories
Write-Host "[2/7] Creating Output Directories..." -ForegroundColor Yellow
cd $projectDir
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "output/multi_agent" | Out-Null
Write-Host "  ✅ Output directories ready" -ForegroundColor Green
Write-Host ""

# Start Kafka (if available)
if ($useKafka) {
    Write-Host "[3/7] Starting Kafka Infrastructure..." -ForegroundColor Yellow
    
    # Check if Kafka already running
    $kafkaProc = Get-Process java -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*kafka*"}
    if ($kafkaProc) {
        Write-Host "  ✅ Kafka already running" -ForegroundColor Green
    } else {
        Write-Host "  🚀 Starting Zookeeper..." -ForegroundColor Cyan
        $zkCmd = "cd '$kafkaDir'; .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $zkCmd
        Start-Sleep -Seconds 5
        
        Write-Host "  🚀 Starting Kafka Broker..." -ForegroundColor Cyan
        $kafkaCmd = "cd '$kafkaDir'; .\bin\windows\kafka-server-start.bat .\config\server.properties"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $kafkaCmd
        Start-Sleep -Seconds 8
        
        Write-Host "  ✅ Kafka infrastructure started" -ForegroundColor Green
    }
} else {
    Write-Host "[3/7] Skipping Kafka (not found)..." -ForegroundColor Yellow
}
Write-Host ""

# Start Producer (if Kafka available)
if ($useKafka) {
    Write-Host "[4/7] Starting Kafka Producer..." -ForegroundColor Yellow
    $prodCmd = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python stream_server_kafka.py --delay 0.05 --loop'
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $prodCmd
    Start-Sleep -Seconds 3
    Write-Host "  ✅ Producer started" -ForegroundColor Green
} else {
    Write-Host "[4/7] Skipping Producer (Kafka not available)..." -ForegroundColor Yellow
}
Write-Host ""

# Start Stream Processor (if Kafka available)
if ($useKafka) {
    Write-Host "[5/7] Starting Stream Processor..." -ForegroundColor Yellow
    $procCmd = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python processor_consumer.py --checkpoint 3'
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $procCmd
    Start-Sleep -Seconds 3
    Write-Host "  ✅ Processor started" -ForegroundColor Green
} else {
    Write-Host "[5/7] Skipping Processor (Kafka not available)..." -ForegroundColor Yellow
}
Write-Host ""

# Start ML Service
Write-Host "[6/7] Starting ML Service (Enhanced with SHAP + LIME)..." -ForegroundColor Yellow
$mlCmd = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python ml_service_enhanced.py'
Start-Process powershell -ArgumentList "-NoExit", "-Command", $mlCmd
Start-Sleep -Seconds 4
Write-Host "  ✅ ML Service started on port 8000" -ForegroundColor Green
Write-Host ""

# Start Multi-Agent System
Write-Host "[7/7] Starting Multi-Agent System..." -ForegroundColor Yellow
$agentCmd = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python multi_agent_system.py --interval 30'
Start-Process powershell -ArgumentList "-NoExit", "-Command", $agentCmd
Start-Sleep -Seconds 3
Write-Host "  ✅ Multi-Agent System started" -ForegroundColor Green
Write-Host ""

# Wait for services to initialize
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Dashboard WITH Agent
Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  STARTING DASHBOARDS" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

Write-Host "[Dashboard 1] Starting WITH Multi-Agent (Port 8501)..." -ForegroundColor Cyan
$dash1Cmd = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; streamlit run dashboard_with_agent.py --server.port 8501'
Start-Process powershell -ArgumentList "-NoExit", "-Command", $dash1Cmd
Start-Sleep -Seconds 5

Write-Host "[Dashboard 2] Starting WITHOUT Agent (Port 8502)..." -ForegroundColor Cyan
$dash2Cmd = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; streamlit run dashboard_without_agent.py --server.port 8502'
Start-Process powershell -ArgumentList "-NoExit", "-Command", $dash2Cmd
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "  ✅ SYSTEM STARTUP COMPLETE!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

Write-Host "📊 Access Your Dashboards:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  🤖 WITH Multi-Agent System:" -ForegroundColor Cyan
Write-Host "     http://localhost:8501" -ForegroundColor White
Write-Host "     - Autonomous decision-making" -ForegroundColor Gray
Write-Host "     - Inventory tracking with stock updates" -ForegroundColor Gray
Write-Host "     - Negative stock prevention" -ForegroundColor Gray
Write-Host "     - LIME explanations for every decision" -ForegroundColor Gray
Write-Host ""
Write-Host "  📋 WITHOUT Agent (Manual):" -ForegroundColor Cyan
Write-Host "     http://localhost:8502" -ForegroundColor White
Write-Host "     - Manual inventory management" -ForegroundColor Gray
Write-Host "     - Traditional approach comparison" -ForegroundColor Gray
Write-Host ""
Write-Host "  🔬 ML Service API:" -ForegroundColor Cyan
Write-Host "     http://localhost:8000" -ForegroundColor White
Write-Host "     http://localhost:8000/docs (API Documentation)" -ForegroundColor White
Write-Host ""

Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Visit http://localhost:8501 (WITH Agent)" -ForegroundColor White
Write-Host "2. Go to Tab 2: Autonomous Agent Dashboard" -ForegroundColor White
Write-Host "3. Watch agents make autonomous decisions" -ForegroundColor White
Write-Host "4. Check inventory tracking in real-time" -ForegroundColor White
Write-Host ""
Write-Host "💡 To generate predictions for agents to process:" -ForegroundColor Yellow
Write-Host "   curl -X POST http://localhost:8000/batch-predict" -ForegroundColor White
Write-Host ""
Write-Host "📁 Output Files:" -ForegroundColor Yellow
Write-Host "   - output/agent_actions_with_lime.csv (All decisions)" -ForegroundColor White
Write-Host "   - output/inventory_state.csv (Current inventory)" -ForegroundColor White
Write-Host "   - output/multi_agent/ (Consensus details)" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C in any service window to stop that service" -ForegroundColor Gray
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
