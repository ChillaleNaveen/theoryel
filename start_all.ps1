Write-Host "Starting LiveInsight+ System" -ForegroundColor Green

$project = "C:\Users\borut\Desktop\main el"
$condaHook = "C:\Users\borut\anaconda3\shell\condabin\conda-hook.ps1"
$condaEnv = "hf-sentiment"

if (-not (Test-Path "$project\output")) {
    New-Item -ItemType Directory -Path "$project\output" | Out-Null
}

Write-Host "Launching 5 services..." -ForegroundColor Cyan

$cmd1 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $project + '"; python stream_server_kafka.py --delay 0.05 --loop'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd1
Start-Sleep -Seconds 3

$cmd2 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $project + '"; python processor_consumer.py --checkpoint 3'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd2
Start-Sleep -Seconds 3

$cmd3 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $project + '"; python ml_service.py'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd3
Start-Sleep -Seconds 3

$cmd4 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $project + '"; python agent.py --interval 30'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd4
Start-Sleep -Seconds 3

$cmd5 = '& "' + $condaHook + '"; conda activate ' + $condaEnv + '; cd "' + $project + '"; streamlit run dashboard_new.py'
Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd5

Write-Host "All services launched!" -ForegroundColor Green
Write-Host "Dashboard: http://localhost:8501" -ForegroundColor Yellow
Write-Host "ML API: http://localhost:8000" -ForegroundColor Yellow
