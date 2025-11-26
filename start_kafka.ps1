# Start Kafka Complete Setup Script
# This script starts Zookeeper and Kafka from C:\kafka1

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "🚀 Starting Kafka Infrastructure" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$kafkaDir = "C:\kafka1"

# Check if Kafka exists
if (-not (Test-Path $kafkaDir)) {
    Write-Host "❌ Kafka not found at $kafkaDir" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Found Kafka at $kafkaDir" -ForegroundColor Green
Write-Host ""

# Clean old data for fresh start
Write-Host "🧹 Cleaning old Kafka/Zookeeper data..." -ForegroundColor Yellow
Remove-Item -Recurse -Force C:\tmp\kafka-logs -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force C:\tmp\zookeeper -ErrorAction SilentlyContinue
Write-Host "✅ Cleaned successfully" -ForegroundColor Green
Write-Host ""

# Start Zookeeper
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "[1/2] Starting Zookeeper..." -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$zookeeperCmd = "cd $kafkaDir; .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties"
Start-Process powershell -ArgumentList "-NoExit","-Command",$zookeeperCmd
Write-Host "✅ Zookeeper started in new window" -ForegroundColor Green
Write-Host "⏳ Waiting 10 seconds for Zookeeper to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Start Kafka Broker
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "[2/2] Starting Kafka Broker..." -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$kafkaCmd = "cd $kafkaDir; .\bin\windows\kafka-server-start.bat .\config\server.properties"
Start-Process powershell -ArgumentList "-NoExit","-Command",$kafkaCmd
Write-Host "✅ Kafka Broker started in new window" -ForegroundColor Green
Write-Host "⏳ Waiting 15 seconds for Kafka to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Create topic
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "📝 Creating Kafka Topic: retail.transactions" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

cd $kafkaDir
.\bin\windows\kafka-topics.bat --create --topic retail.transactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Topic 'retail.transactions' created successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  Topic might already exist (this is OK)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "✅ Kafka Infrastructure Ready!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Keep the Zookeeper and Kafka windows open" -ForegroundColor Gray
Write-Host "   2. Run: .\start_all.ps1" -ForegroundColor Gray
Write-Host ""
