# LiveInsight+ Dual Dashboard Launcher
# Starts BOTH dashboards for side-by-side comparison
# Dashboard WITH Agent (Port 8501) vs Dashboard WITHOUT Agent (Port 8502)

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "DUAL DASHBOARD COMPARISON LAUNCHER" -ForegroundColor Green
Write-Host "Side-by-Side: Autonomous AI vs Manual Process" -ForegroundColor Yellow
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$projectDir = "C:\Users\borut\Desktop\7 th sem el\retail-with-agent-and-explainable-ai"
$condaHook = "C:\Users\borut\anaconda3\shell\condabin\conda-hook.ps1"
$condaEnv = "hf-sentiment"

Write-Host "Launching comparison dashboards..." -ForegroundColor Yellow
Write-Host ""

# Dashboard WITH Agent (Port 8501)
Write-Host "[1/2] Starting Dashboard WITH LIME Agent (Port 8501)..." -ForegroundColor Cyan
$cmd1 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; streamlit run dashboard_with_agent.py --server.port 8501'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd1
Write-Host "  Autonomous AI Dashboard started" -ForegroundColor Green
Start-Sleep -Seconds 3

# Dashboard WITHOUT Agent (Port 8502)
Write-Host "[2/2] Starting Dashboard WITHOUT Agent (Port 8502)..." -ForegroundColor Cyan
$cmd2 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $projectDir + '"; streamlit run dashboard_without_agent.py --server.port 8502'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd2
Write-Host "  Manual Process Dashboard started" -ForegroundColor Green

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "BOTH DASHBOARDS READY FOR COMPARISON!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Access Points:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  WITH Agent (Autonomous):  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8501" -ForegroundColor Cyan
Write-Host "    -> LIME-powered autonomous decisions" -ForegroundColor Gray
Write-Host "    -> Real-time agent actions with explanations" -ForegroundColor Gray
Write-Host "    -> Auto-approval workflow" -ForegroundColor Gray
Write-Host ""
Write-Host "  WITHOUT Agent (Manual):   " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8502" -ForegroundColor Cyan
Write-Host "    -> Manual inventory review required" -ForegroundColor Gray
Write-Host "    -> Time-consuming human decisions" -ForegroundColor Gray
Write-Host "    -> No automation" -ForegroundColor Gray
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Open BOTH URLs in your browser to compare side-by-side!" -ForegroundColor Yellow
Write-Host ""
Write-Host "To Stop: Get-Process streamlit | Stop-Process" -ForegroundColor Gray
Write-Host ""
