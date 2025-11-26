"""
System Health Monitor for LiveInsight+
Provides health check endpoints and system metrics
Can be queried by dashboard for monitoring
"""

import os
import psutil
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="LiveInsight+ Health Monitor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LiveInsight+ Health Monitor",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('C:\\' if os.name == 'nt' else '/')
        net = psutil.net_io_counters()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "network": {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv,
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv
            },
            "process": {
                "pid": os.getpid(),
                "threads": psutil.Process().num_threads(),
                "memory_mb": psutil.Process().memory_info().rss / (1024**2)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/metrics")
async def get_metrics():
    """Get detailed system metrics"""
    try:
        # CPU metrics
        cpu_freq = psutil.cpu_freq()
        cpu_stats = psutil.cpu_stats()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.5),
                "count": psutil.cpu_count(),
                "freq_current": cpu_freq.current if cpu_freq else 0,
                "freq_max": cpu_freq.max if cpu_freq else 0,
                "ctx_switches": cpu_stats.ctx_switches,
                "interrupts": cpu_stats.interrupts
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
                "swap_percent": swap.percent
            },
            "disk": {
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/temperature")
async def get_temperature():
    """Get system temperature (if available)"""
    temps = {}
    
    try:
        if hasattr(psutil, "sensors_temperatures"):
            sensors = psutil.sensors_temperatures()
            for name, entries in sensors.items():
                temps[name] = [
                    {
                        "label": entry.label,
                        "current": entry.current,
                        "high": entry.high,
                        "critical": entry.critical
                    }
                    for entry in entries
                ]
    except Exception as e:
        temps["error"] = str(e)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "temperatures": temps
    }


@app.get("/processes")
async def get_top_processes(limit: int = 10):
    """Get top processes by CPU and memory"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU
        top_cpu = sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:limit]
        
        # Sort by memory
        top_mem = sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:limit]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "top_cpu": top_cpu,
            "top_memory": top_mem
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("🏥 Starting Health Monitor on port 9000")
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
