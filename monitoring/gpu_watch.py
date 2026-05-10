import subprocess
import json
import time
import datetime

def get_gpu_stats():
    """
    Ask nvidia-smi for structured GPU data.
    We use --format=csv,noheader,nounits so we get clean numbers.
    """
    result = subprocess.run([
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ], capture_output=True, text=True)
    
    line = result.stdout.strip()
    parts = [p.strip() for p in line.split(",")]
    
    return {
        "timestamp": parts[0],
        "gpu_name": parts[1],
        "vram_used_mb": int(parts[2]),
        "vram_free_mb": int(parts[3]),
        "vram_total_mb": int(parts[4]),
        "gpu_util_pct": int(parts[5]),
        "temp_celsius": int(parts[6]),
        "power_draw_w": float(parts[7]),
        "vram_used_pct": round(int(parts[2]) / int(parts[4]) * 100, 1)
    }

def watch(interval_sec=1.0, output_file=None):
    """
    Poll GPU stats every interval_sec seconds.
    Print to console AND optionally write to a JSON file for the dashboard.
    """
    print(f"{'Time':12} {'VRAM Used':>12} {'VRAM %':>8} {'GPU Util':>10} {'Temp':>6} {'Power':>8}")
    print("-" * 60)
    
    log = []
    
    try:
        while True:
            stats = get_gpu_stats()
            log.append(stats)
            
            print(
                f"{datetime.datetime.now().strftime('%H:%M:%S'):12}"
                f"{stats['vram_used_mb']:>9} MB"
                f"{stats['vram_used_pct']:>8}%"
                f"{stats['gpu_util_pct']:>9}%"
                f"{stats['temp_celsius']:>5}°C"
                f"{stats['power_draw_w']:>7.1f}W"
            )
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(log, f, indent=2)
            
            time.sleep(interval_sec)
            
    except KeyboardInterrupt:
        print(f"\nStopped. Captured {len(log)} data points.")
        if output_file:
            print(f"Data saved to {output_file}")

if __name__ == "__main__":
    watch(interval_sec=1.0, output_file="benchmarks/results/gpu_log.json")