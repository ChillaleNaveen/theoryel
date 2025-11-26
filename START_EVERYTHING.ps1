# ULTIMATE SYSTEM LAUNCHER
# Starts complete system with dual dashboards for comparison
# Includes: Kafka + Producer + Processor + ML + Agent + BOTH Dashboards

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "ULTIMATE LIVEINSIGHT+ SYSTEM LAUNCHER" -ForegroundColor Green
Write-Host "Complete System with Side-by-Side Comparison Dashboards" -ForegroundColor Yellow
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$kafkaDir = "C:\kafka1"
$projectDir = "C:\Users\borut\Desktop\7 th sem el\retail-with-agent-and-explainable-ai"
$condaHook = "C:\Users\borut\anaconda3\shell\condabin\conda-hook.ps1"
$condaEnv = "hf-sentiment"

# Check prerequisites
Write-Host "[1/8] Checking prerequisites..." -ForegroundColor Yellow
if (-not (Test-Path $kafkaDir)) {
    Write-Host "Error: Kafka not found at $kafkaDir" -ForegroundColor Red
    exit 1
}
Write-Host "  Kafka found" -ForegroundColor Green
Write-Host "  Python/Conda configured" -ForegroundColor Green
Write-Host ""

# Create directories
Write-Host "[2/8] Creating output directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$projectDir\output" | Out-Null
New-Item -ItemType Directory -Force -Path "$projectDir\output\xai\shap" | Out-Null
New-Item -ItemType Directory -Force -Path "$projectDir\output\xai\lime" | Out-Null
New-Item -ItemType Directory -Force -Path "$projectDir\models" | Out-Null
Write-Host "  Directories created" -ForegroundColor Green
Write-Host ""

# Clean Kafka data
Write-Host "[3/8] Cleaning old Kafka data..." -ForegroundColor Yellow
Remove-Item -Recurse -Force "C:\tmp\kafka-logs" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "C:\tmp\zookeeper" -ErrorAction SilentlyContinue
Write-Host "  Cleaned" -ForegroundColor Green
Write-Host ""

# Start Zookeeper
Write-Host "[4/8] Starting Kafka infrastructure..." -ForegroundColor Yellow
Write-Host "  [4.1] Starting Zookeeper..." -ForegroundColor Cyan
$zookeeperCmd = "cd $kafkaDir; .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties"
Start-Process powershell -ArgumentList "-NoExit","-Command",$zookeeperCmd
Start-Sleep -Seconds 10

# Start Kafka
Write-Host "  [4.2] Starting Kafka Broker..." -ForegroundColor Cyan
$kafkaCmd = "cd $kafkaDir; .\bin\windows\kafka-server-start.bat .\config\server.properties"
Start-Process powershell -ArgumentList "-NoExit","-Command",$kafkaCmd
Start-Sleep -Seconds 15

# Create topic
Write-Host "  [4.3] Creating Kafka topic..." -ForegroundColor Cyan
cd $kafkaDir
.\bin\windows\kafka-topics.bat --create --topic retail.transactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists 2>$null
Write-Host "  Kafka infrastructure ready" -ForegroundColor Green
Write-Host ""

cd $projectDir

# Start data services
Write-Host "[5/8] Starting data services..." -ForegroundColor Yellow

Write-Host "  [5.1] Kafka Producer (randomized)..." -ForegroundColor Cyan
$cmd1 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python stream_server_kafka.py --delay 0.03 --loop'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd1
Start-Sleep -Seconds 3

Write-Host "  [5.2] Stream Processor..." -ForegroundColor Cyan
$cmd2 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python processor_consumer.py --checkpoint 3'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd2
Start-Sleep -Seconds 3

Write-Host "  Data services running" -ForegroundColor Green
Write-Host ""

# Start ML service
Write-Host "[6/8] Starting Enhanced ML Service (SHAP + LIME)..." -ForegroundColor Yellow
$cmd3 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python ml_service_enhanced.py'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd3
Write-Host "  ML service running on port 8000" -ForegroundColor Green
Start-Sleep -Seconds 5
Write-Host ""

# Start GOD-TIER Multi-Agent System
Write-Host "[7/8] Starting GOD-TIER Multi-Agent System with LIME..." -ForegroundColor Yellow
Write-Host "  🌟 Initializing 3 specialized agents:" -ForegroundColor Magenta
Write-Host "     🚨 Urgency Agent - Emergency detection" -ForegroundColor Magenta
Write-Host "     📦 Quantity Agent - Optimal order sizing" -ForegroundColor Magenta
Write-Host "     💰 Cost Agent - Cost-benefit analysis" -ForegroundColor Magenta
$cmd4 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; python multi_agent_system.py --interval 30'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd4
Write-Host "  ✅ Multi-Agent System ACTIVE - Fully autonomous decisions" -ForegroundColor Green
Start-Sleep -Seconds 3
Write-Host ""

# Start BOTH dashboards
Write-Host "[8/8] Starting Comparison Dashboards..." -ForegroundColor Yellow

Write-Host "  [8.1] Dashboard WITH Agent (Port 8501)..." -ForegroundColor Cyan
$cmd5 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; streamlit run dashboard_with_agent.py --server.port 8501'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd5
Start-Sleep -Seconds 3

Write-Host "  [8.2] Dashboard WITHOUT Agent (Port 8502)..." -ForegroundColor Cyan
$cmd6 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; streamlit run dashboard_without_agent.py --server.port 8502'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd6

Write-Host "  Both dashboards launched" -ForegroundColor Green
Write-Host ""

Write-Host "Waiting for initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "COMPLETE SYSTEM RUNNING WITH DUAL DASHBOARDS!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Open Windows:" -ForegroundColor Yellow
Write-Host "  1. Zookeeper" -ForegroundColor Gray
Write-Host "  2. Kafka Broker" -ForegroundColor Gray
Write-Host "  3. Kafka Producer (randomized streaming)" -ForegroundColor Gray
Write-Host "  4. Stream Processor" -ForegroundColor Gray
Write-Host "  5. ML Service (SHAP + LIME)" -ForegroundColor Gray
Write-Host "  6. LIME-Powered Agent" -ForegroundColor Gray
Write-Host "  7. Dashboard WITH Agent" -ForegroundColor Gray
Write-Host "  8. Dashboard WITHOUT Agent" -ForegroundColor Gray
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "ACCESS POINTS FOR COMPARISON:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Dashboard WITH Agent (Autonomous AI):" -ForegroundColor White
Write-Host "  http://localhost:8501" -ForegroundColor Cyan -NoNewline
Write-Host "  <- LIME agent with transparent decisions" -ForegroundColor Gray
Write-Host ""
Write-Host "Dashboard WITHOUT Agent (Manual Process):" -ForegroundColor White
Write-Host "  http://localhost:8502" -ForegroundColor Cyan -NoNewline
Write-Host "  <- Traditional manual inventory management" -ForegroundColor Gray
Write-Host ""
Write-Host "ML API Service:" -ForegroundColor White
Write-Host "  http://localhost:8000" -ForegroundColor Cyan -NoNewline
Write-Host "  <- Random Forest + SHAP + LIME" -ForegroundColor Gray
Write-Host ""
Write-Host "API Documentation:" -ForegroundColor White
Write-Host "  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "KEY COMPARISON POINTS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "WITH Agent (8501):" -ForegroundColor Green
Write-Host "  - Autonomous decisions every 30 seconds" -ForegroundColor Gray
Write-Host "  - LIME explanations for transparency" -ForegroundColor Gray
Write-Host "  - Auto-approval for high-confidence actions" -ForegroundColor Gray
Write-Host "  - Complete audit trail" -ForegroundColor Gray
Write-Host "  - 24/7 monitoring capability" -ForegroundColor Gray
Write-Host ""
Write-Host "WITHOUT Agent (8502):" -ForegroundColor Red
Write-Host "  - Manual review required (5-30 min per product)" -ForegroundColor Gray
Write-Host "  - Human decision-making delays" -ForegroundColor Gray
Write-Host "  - Risk of oversight" -ForegroundColor Gray
Write-Host "  - Limited to business hours" -ForegroundColor Gray
Write-Host "  - Inconsistent responses" -ForegroundColor Gray
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "To Stop All Services:" -ForegroundColor Yellow
Write-Host "  Get-Process python,java,streamlit | Stop-Process -Force" -ForegroundColor Gray
Write-Host ""
