# Complete System Launcher - LiveInsight+ WITH Randomized Real-Time Streaming
# This script starts EVERYTHING including Kafka infrastructure
# PowerShell script for Windows

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "🚀 LiveInsight+ COMPLETE SYSTEM LAUNCHER" -ForegroundColor Green
Write-Host "   WITH Randomized Real-Time Data Streaming Simulation" -ForegroundColor Yellow
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Configuration
$kafkaDir = "C:\kafka1"
$projectDir = "C:\Users\borut\Desktop\7 th sem el\retail-with-agent-and-explainable-ai"
$condaHook = "C:\Users\borut\anaconda3\shell\condabin\conda-hook.ps1"
$condaEnv = "hf-sentiment"

# Step 1: Check Prerequisites
Write-Host "[1/7] Checking Prerequisites..." -ForegroundColor Yellow
Write-Host ""

# Check Kafka
if (-not (Test-Path $kafkaDir)) {
    Write-Host "❌ Kafka not found at $kafkaDir" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Kafka found at $kafkaDir" -ForegroundColor Green

# Check Python/Conda
if (-not (Test-Path $condaHook)) {
    Write-Host "⚠️  Conda not found, using system Python" -ForegroundColor Yellow
    $useConda = $false
} else {
    Write-Host "✅ Conda found" -ForegroundColor Green
    $useConda = $true
}

# Check project directory
if (-not (Test-Path $projectDir)) {
    Write-Host "❌ Project directory not found: $projectDir" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Project directory found" -ForegroundColor Green
Write-Host ""

# Step 2: Clean old data
Write-Host "[2/7] Cleaning old Kafka/Zookeeper data..." -ForegroundColor Yellow
Remove-Item -Recurse -Force "C:\tmp\kafka-logs" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "C:\tmp\zookeeper" -ErrorAction SilentlyContinue
Write-Host "✅ Cleaned successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Create output directories
Write-Host "[3/7] Creating output directories..." -ForegroundColor Yellow
$outputDirs = @(
    "$projectDir\output",
    "$projectDir\output\xai",
    "$projectDir\output\xai\shap",
    "$projectDir\output\xai\lime",
    "$projectDir\models"
)
foreach ($dir in $outputDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "✅ Directories created" -ForegroundColor Green
Write-Host ""

# Step 4: Start Kafka Infrastructure
Write-Host "[4/7] Starting Kafka Infrastructure..." -ForegroundColor Yellow
Write-Host ""

# Start Zookeeper
Write-Host "  [4.1] Starting Zookeeper..." -ForegroundColor Cyan
$zookeeperCmd = "cd $kafkaDir; .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties"
Start-Process powershell -ArgumentList "-NoExit","-Command",$zookeeperCmd
Write-Host "  ✅ Zookeeper started" -ForegroundColor Green
Write-Host "  ⏳ Waiting 10 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 10

# Start Kafka Broker
Write-Host "  [4.2] Starting Kafka Broker..." -ForegroundColor Cyan
$kafkaCmd = "cd $kafkaDir; .\bin\windows\kafka-server-start.bat .\config\server.properties"
Start-Process powershell -ArgumentList "-NoExit","-Command",$kafkaCmd
Write-Host "  ✅ Kafka Broker started" -ForegroundColor Green
Write-Host "  ⏳ Waiting 15 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 15

# Create Kafka Topic
Write-Host "  [4.3] Creating Kafka topic..." -ForegroundColor Cyan
cd $kafkaDir
.\bin\windows\kafka-topics.bat --create --topic retail.transactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists 2>$null
Write-Host "  ✅ Topic 'retail.transactions' ready" -ForegroundColor Green
Write-Host ""

# Step 5: Start Python Services
Write-Host "[5/7] Starting Python Services..." -ForegroundColor Yellow
Write-Host ""

cd $projectDir

if ($useConda) {
    # With Conda
    # Service 1: Kafka Producer with Randomization
    Write-Host "  [5.1] Starting Randomized Kafka Producer..." -ForegroundColor Cyan
    $cmd1 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python stream_server_kafka.py --delay 0.03 --loop'
    Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd1
    Write-Host "  ✅ Producer started (randomized mode)" -ForegroundColor Green
    Start-Sleep -Seconds 3

    # Service 2: Stream Processor
    Write-Host "  [5.2] Starting Stream Processor..." -ForegroundColor Cyan
    $cmd2 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python processor_consumer.py --checkpoint 3'
    Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd2
    Write-Host "  ✅ Processor started" -ForegroundColor Green
    Start-Sleep -Seconds 3

    # Service 3: Enhanced ML Service with LIME
    Write-Host "  [5.3] Starting Enhanced ML Service (SHAP + LIME)..." -ForegroundColor Cyan
    $cmd3 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python ml_service_enhanced.py'
    Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd3
    Write-Host "  ✅ ML Service started" -ForegroundColor Green
    Start-Sleep -Seconds 5

    # Service 4: Agent with LIME
    Write-Host "  [5.4] Starting Autonomous Agent (LIME-powered)..." -ForegroundColor Cyan
    $cmd4 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python agent_with_lime.py --interval 30'
    Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd4
    Write-Host "  ✅ Agent started" -ForegroundColor Green
    Start-Sleep -Seconds 3

    # Service 5: Dashboard WITH Agent
    Write-Host "  [5.5] Starting Dashboard WITH Agent (Port 8501)..." -ForegroundColor Cyan
    $cmd5 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; streamlit run dashboard_with_agent.py'
    Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd5
    Write-Host "  ✅ Dashboard started" -ForegroundColor Green
} else {
    # Without Conda (system Python)
    Write-Host "  [5.1] Starting Randomized Kafka Producer..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$projectDir'; python stream_server_kafka.py --delay 0.03 --loop"
    Write-Host "  ✅ Producer started (randomized mode)" -ForegroundColor Green
    Start-Sleep -Seconds 3

    Write-Host "  [5.2] Starting Stream Processor..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$projectDir'; python processor_consumer.py --checkpoint 3"
    Write-Host "  ✅ Processor started" -ForegroundColor Green
    Start-Sleep -Seconds 3

    Write-Host "  [5.3] Starting Enhanced ML Service (SHAP + LIME)..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$projectDir'; python ml_service_enhanced.py"
    Write-Host "  ✅ ML Service started" -ForegroundColor Green
    Start-Sleep -Seconds 5

    Write-Host "  [5.4] Starting Autonomous Agent (LIME-powered)..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$projectDir'; python agent_with_lime.py --interval 30"
    Write-Host "  ✅ Agent started" -ForegroundColor Green
    Start-Sleep -Seconds 3

    Write-Host "  [5.5] Starting Dashboard WITH Agent (Port 8501)..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$projectDir'; streamlit run dashboard_with_agent.py"
    Write-Host "  ✅ Dashboard started" -ForegroundColor Green
}

Write-Host ""

# Step 6: Wait for initialization
Write-Host "[6/7] Waiting for services to initialize..." -ForegroundColor Yellow
Write-Host "⏳ Please wait 10 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 10
Write-Host "✅ System should be ready" -ForegroundColor Green
Write-Host ""

# Step 7: Display Access Information
Write-Host "[7/7] System Ready!" -ForegroundColor Green
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "✅ ALL SERVICES RUNNING" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 Access Points:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  📊 Dashboard WITH Agent:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8501" -ForegroundColor Cyan
Write-Host "     → Autonomous AI-driven inventory management" -ForegroundColor Gray
Write-Host "     → Real-time LIME explanations for agent decisions" -ForegroundColor Gray
Write-Host "     → Performance metrics and comparison" -ForegroundColor Gray
Write-Host ""
Write-Host "  🔬 ML API Service:         " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host "     → Random Forest predictions" -ForegroundColor Gray
Write-Host "     → SHAP global explanations" -ForegroundColor Gray
Write-Host "     → LIME agent decision explanations" -ForegroundColor Gray
Write-Host ""
Write-Host "  📚 API Documentation:      " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 Key Features:" -ForegroundColor Yellow
Write-Host "  ✅ Randomized real-time data streaming (simulates live retail)" -ForegroundColor Green
Write-Host "  ✅ Kafka streaming with stateful processing" -ForegroundColor Green
Write-Host "  ✅ Random Forest ML with SHAP + LIME explanations" -ForegroundColor Green
Write-Host "  ✅ Autonomous agent with transparent LIME decisions" -ForegroundColor Green
Write-Host "  ✅ Interactive dashboard with real-time analytics" -ForegroundColor Green
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Open Windows:" -ForegroundColor Yellow
Write-Host "  • Zookeeper" -ForegroundColor Gray
Write-Host "  • Kafka Broker" -ForegroundColor Gray
Write-Host "  • Kafka Producer (randomized)" -ForegroundColor Gray
Write-Host "  • Stream Processor" -ForegroundColor Gray
Write-Host "  • ML Service" -ForegroundColor Gray
Write-Host "  • Autonomous Agent" -ForegroundColor Gray
Write-Host "  • Dashboard" -ForegroundColor Gray
Write-Host ""
Write-Host "🛑 To Stop All Services:" -ForegroundColor Yellow
Write-Host "   Get-Process python,java,streamlit | Stop-Process -Force" -ForegroundColor Gray
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Enjoy your AI-powered retail intelligence system!" -ForegroundColor Green
Write-Host ""
