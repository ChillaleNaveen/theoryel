#!/bin/bash

# LiveInsight+ Startup Script (Linux/Mac)
# Run this to start all services in separate terminal tabs

echo "🚀 Starting LiveInsight+ System"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.9+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if required files exist
required_files=(
    "stream_server_kafka.py"
    "processor_consumer.py"
    "ml_service.py"
    "agent.py"
    "dashboard_new.py"
    "retail_data_bangalore.csv"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        exit 1
    fi
done

echo "✅ All required files present"
echo ""

# Create output directory
mkdir -p output
echo "📁 Created output directory"
echo ""

echo "🎯 Launching services..."
echo ""

# Detect terminal emulator
if command -v gnome-terminal &> /dev/null; then
    TERM_CMD="gnome-terminal"
    TAB_ARG="--tab"
elif command -v konsole &> /dev/null; then
    TERM_CMD="konsole"
    TAB_ARG="--new-tab"
elif command -v xterm &> /dev/null; then
    TERM_CMD="xterm"
    TAB_ARG="-e"
else
    echo "⚠️  No supported terminal found. Launching in background..."
    echo ""
    
    # Background mode
    python3 stream_server_kafka.py --delay 0.05 --loop > logs/producer.log 2>&1 &
    echo "1️⃣  Kafka Producer started (PID: $!)"
    
    python3 processor_consumer.py --checkpoint 3 > logs/processor.log 2>&1 &
    echo "2️⃣  Stream Processor started (PID: $!)"
    
    python3 ml_service.py > logs/ml_service.log 2>&1 &
    echo "3️⃣  ML Service started (PID: $!)"
    
    python3 agent.py --interval 30 > logs/agent.log 2>&1 &
    echo "4️⃣  Agent started (PID: $!)"
    
    sleep 3
    streamlit run dashboard_new.py
    
    exit 0
fi

# Terminal mode (if supported terminal found)
$TERM_CMD \
    $TAB_ARG --title="Producer" -- bash -c "python3 stream_server_kafka.py --delay 0.05 --loop; exec bash" \
    $TAB_ARG --title="Processor" -- bash -c "sleep 2; python3 processor_consumer.py --checkpoint 3; exec bash" \
    $TAB_ARG --title="ML Service" -- bash -c "sleep 4; python3 ml_service.py; exec bash" \
    $TAB_ARG --title="Agent" -- bash -c "sleep 6; python3 agent.py --interval 30; exec bash" \
    $TAB_ARG --title="Dashboard" -- bash -c "sleep 8; streamlit run dashboard_new.py; exec bash" &

echo ""
echo "================================"
echo "✅ All services launched!"
echo ""
echo "📊 Dashboard will open at: http://localhost:8501"
echo "🔬 ML API available at: http://localhost:8000"
echo ""
echo "⚠️  To stop all services:"
echo "   pkill -f 'python3.*kafka|processor|ml_service|agent'"
echo "   pkill -f streamlit"
echo ""
echo "📖 See RUNBOOK.md for detailed instructions"
echo ""
