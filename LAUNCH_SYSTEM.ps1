# ============================================================
# LiveInsight+ Launcher (Hard Reset Edition)
# ============================================================

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "  LIVEINSIGHT+ SYSTEM LAUNCHER (AUTO-RECOVERY)"
Write-Host "  Includes Zookeeper/Kafka Hard Reset"
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

# -----------------------------
# CONFIGURATION
# -----------------------------
$projectDir = "C:\Users\cnave\OneDrive\Desktop\retail-with-agent-and-explainable-ai"
$kafkaDir = "C:\kafka"
$pythonVenv = "$projectDir\.venv\Scripts\python.exe"
$activateVenv = "$projectDir\.venv\Scripts\Activate.ps1"
$tmpDir = "C:\tmp"

# -----------------------------
# 0. HARD RESET (The Fix)
# -----------------------------
Write-Host "[0/7] Performing System Cleanup..." -ForegroundColor Yellow

# 1. Kill any stuck Java processes (Fixes 'Address already in use')
try {
    Write-Host "   - Stopping stuck Java/Kafka processes..."
    Stop-Process -Name "java" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
} catch {
    Write-Host "   - No stuck Java processes found."
}

# 2. Delete Kafka Temp Logs (Fixes 'NodeExists' / Crash on startup)
if (Test-Path $tmpDir) {
    Write-Host "   - Clearing Kafka temporary data (C:\tmp)..."
    try {
        Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction Stop
        Write-Host "   - Cleanup Successful." -ForegroundColor Green
    } catch {
        Write-Host "   - WARNING: Could not delete C:\tmp. Ensure no files are open." -ForegroundColor Red
    }
} else {
    Write-Host "   - C:\tmp is already clean."
}

# -----------------------------
# 1. Prerequisite Check
# -----------------------------
Write-Host "[1/7] Checking environment..."

if (!(Test-Path $pythonVenv)) {
    Write-Host "ERROR: Python virtual environment NOT found at $pythonVenv" -ForegroundColor Red
    exit 1
} else {
    Write-Host "OK: Python Venv detected."
}

if (Test-Path $kafkaDir) {
    Write-Host "OK: Kafka detected."
    $useKafka = $true
} else {
    Write-Host "WARNING: Kafka NOT found. Running without streaming." -ForegroundColor Red
    $useKafka = $false
}

# -----------------------------
# 2. Output Directory Setup
# -----------------------------
Write-Host "[2/7] Creating output directories..."
New-Item -ItemType Directory -Force -Path "$projectDir/output" | Out-Null
New-Item -ItemType Directory -Force -Path "$projectDir/output/multi_agent" | Out-Null
Write-Host "Output folders ready."

# -----------------------------
# 3. Start Kafka
# -----------------------------
if ($useKafka) {
    Write-Host "[3/7] Starting Kafka Infrastructure..."

    # Start Zookeeper
    Write-Host "   - Launching Zookeeper..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$kafkaDir'; .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties"
    
    # WAIT LONGER for Zookeeper to initialize fully (prevents connection refusal)
    Write-Host "     Waiting 10s for Zookeeper..."
    Start-Sleep -Seconds 10

    # Start Kafka
    Write-Host "   - Launching Kafka Server..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$kafkaDir'; .\bin\windows\kafka-server-start.bat .\config\server.properties"
    
    Write-Host "     Waiting 10s for Kafka..."
    Start-Sleep -Seconds 10
}

# -----------------------------
# 4. ML Service (Must start early)
# -----------------------------
Write-Host "[4/7] Starting ML Service..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir'; $activateVenv; python ml_service_enhanced.py"
Start-Sleep -Seconds 5

# -----------------------------
# 5. Kafka Consumer & Multi-Agent
# -----------------------------
if ($useKafka) {
    Write-Host "[5/7] Starting Consumer & Agents..."
    # Consumer
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir'; $activateVenv; python processor_consumer.py --checkpoint 3"
    Start-Sleep -Seconds 2
    
    # Multi-Agent
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir'; $activateVenv; python multi_agent_system.py --interval 30"
    Start-Sleep -Seconds 2
}
# -----------------------------
# 6. Dashboards
# -----------------------------
Write-Host "[6/7] Starting Dashboards..."

# Dashboard WITH Agent (Port 8501)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir'; $activateVenv; streamlit run dashboard_with_agent.py --server.port 8501"
Start-Sleep -Seconds 2

# Dashboard WITHOUT Agent (Port 8502) - ADDED BACK
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir'; $activateVenv; streamlit run dashboard_without_agent.py --server.port 8502"
# -----------------------------
# 7. Kafka Producer (Start LAST)
# -----------------------------
# We start this last so the Consumer/Dashboards are ready to receive the first messages
if ($useKafka) {
    Write-Host "[7/7] Starting Data Stream (Producer)..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir'; $activateVenv; python stream_server_kafka.py --broker 127.0.0.1:9092 --delay 0.05 --loop"
}

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host " SYSTEM LAUNCHED SUCCESSFULLY"
Write-Host ("=" * 80) -ForegroundColor Green

Write-Host ""
Write-Host "Access your dashboards:"
Write-Host " - http://localhost:8501 (With Agent)"
Write-Host " - http://localhost:8502 (Without Agent)"
Write-Host ""
Write-Host "API available at:"
Write-Host " - http://localhost:8000/docs"
Write-Host ""
Write-Host ("=" * 80)
Write-Host "Press CTRL + C in any opened window to stop services."
Write-Host ("=" * 80)