"""
llama.cpp Benchmark Script — Phase 1
=====================================
What this does:
  Runs llama.cpp with different configurations, captures tokens/sec,
  VRAM usage, and latency, saves everything as structured JSON.

Why structured JSON:
  Every backend (Ollama, vLLM) will output the same JSON schema.
  The dashboard reads one unified format regardless of backend.

How to run:
  python3 benchmark.py --binary ~/projects/llm-fine-tuning/llama.cpp/build/bin/llama-cli
                       --model ~/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf
                       --output ~/projects/llm-inference-core/benchmarks/results/llamacpp.json
"""

import subprocess
import json
import time
import re
import datetime
import argparse
import threading
from pathlib import Path


# ─────────────────────────────────────────────
# GPU SAMPLER
# Runs in a background thread while inference
# is happening. Samples VRAM every 0.5s.
# This gives us the VRAM-over-time curve, not
# just a single snapshot.
# ─────────────────────────────────────────────

class GPUSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def _sample(self):
        """Ask nvidia-smi for one reading."""
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return None

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "ts": time.time(),
            "vram_used_mb": int(parts[0]),
            "gpu_util_pct": int(parts[1]),
            "power_w": float(parts[2]),
            "temp_c": int(parts[3])
        }

    def start(self):
        """Start background sampling thread."""
        def _loop():
            while not self._stop.is_set():
                s = self._sample()
                if s:
                    self.samples.append(s)
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def summary(self):
        """
        Compute peak, average, and delta VRAM from samples.
        Delta = peak - first sample = how much VRAM the model consumed.
        """
        if not self.samples:
            return {}

        vram_values = [s["vram_used_mb"] for s in self.samples]
        util_values = [s["gpu_util_pct"] for s in self.samples]

        return {
            "vram_baseline_mb": vram_values[0],
            "vram_peak_mb": max(vram_values),
            "vram_delta_mb": max(vram_values) - vram_values[0],
            "vram_samples": vram_values,
            "gpu_util_avg_pct": round(sum(util_values) / len(util_values), 1),
            "gpu_util_peak_pct": max(util_values),
            "sample_count": len(self.samples)
        }


# ─────────────────────────────────────────────
# OUTPUT PARSER
# llama.cpp prints timing in this format:
#   [ Prompt: 191.2 t/s | Generation: 89.8 t/s ]
# We also parse the memory breakdown line:
#   | CUDA0 ... | 8150 = 3177 + (3839 = 2228 + 1536 + 74) + 1134 |
# ─────────────────────────────────────────────

def parse_llamacpp_output(output: str) -> dict:
    result = {}

    # Parse tokens/sec line
    # Example: [ Prompt: 191.2 t/s | Generation: 89.8 t/s ]
    timing_match = re.search(
        r'Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s',
        output
    )
    if timing_match:
        result["prompt_tokens_per_sec"] = float(timing_match.group(1))
        result["generation_tokens_per_sec"] = float(timing_match.group(2))

    # Parse memory breakdown line
    # Example: | CUDA0 ... | 8150 = 3177 + (3839 = 2228 + 1536 + 74) + 1134 |
    mem_match = re.search(
        r'CUDA0.*?(\d+)\s*=\s*(\d+)\s*\+\s*\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)',
        output
    )
    if mem_match:
        result["vram_total_mb"] = int(mem_match.group(1))
        result["vram_free_mb"] = int(mem_match.group(2))
        result["vram_used_mb"] = int(mem_match.group(3))
        result["vram_model_mb"] = int(mem_match.group(4))
        result["vram_kvcache_mb"] = int(mem_match.group(5))
        result["vram_compute_mb"] = int(mem_match.group(6))

    return result


# ─────────────────────────────────────────────
# SINGLE RUN
# Runs llama.cpp once with given config.
# Returns a result dict with all metrics.
# ─────────────────────────────────────────────

def run_single(binary: str, model: str, config: dict) -> dict:
    """
    config keys:
      ngl       — number of GPU layers (0 = CPU only, 33 = full GPU)
      n_tokens  — tokens to generate
      prompt    — input text
      temp      — temperature (0 = deterministic)
      label     — human-readable name for this config
    """
    cmd = [
        binary,
        "-m", model,
        "-n", str(config["n_tokens"]),
        "-ngl", str(config["ngl"]),
        "-p", config["prompt"],
        "--temp", str(config.get("temp", 0)),
        "--single-turn",
        "--no-display-prompt"
    ]

    print(f"\n{'='*60}")
    print(f"Running: {config['label']}")
    print(f"  GPU layers: {config['ngl']}")
    print(f"  Tokens to generate: {config['n_tokens']}")
    print(f"{'='*60}")

    # Start GPU sampling before launching inference
    sampler = GPUSampler(interval=0.5)
    sampler.start()

    start_time = time.time()

    proc = subprocess.run(
        cmd,
        capture_output=False,          # let output stream to terminal
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    elapsed = time.time() - start_time
    sampler.stop()

    # Parse llama.cpp output
    parsed = parse_llamacpp_output(proc.stdout)

    # Combine everything into one result record
    result = {
        "label": config["label"],
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "ngl": config["ngl"],
            "n_tokens": config["n_tokens"],
            "prompt": config["prompt"],
            "temp": config.get("temp", 0),
            "model": Path(model).name
        },
        "timing": {
            "wall_time_sec": round(elapsed, 2),
            "prompt_tokens_per_sec": parsed.get("prompt_tokens_per_sec"),
            "generation_tokens_per_sec": parsed.get("generation_tokens_per_sec"),
        },
        "memory": {
            "vram_model_mb": parsed.get("vram_model_mb"),
            "vram_kvcache_mb": parsed.get("vram_kvcache_mb"),
            "vram_compute_mb": parsed.get("vram_compute_mb"),
            "vram_total_used_mb": parsed.get("vram_used_mb"),
            **sampler.summary()
        },
        "raw_output": proc.stdout   # keep full output for debugging
    }

    # Print summary to terminal
    gen = parsed.get("generation_tokens_per_sec", "N/A")
    prompt = parsed.get("prompt_tokens_per_sec", "N/A")
    vram_delta = sampler.summary().get("vram_delta_mb", "N/A")
    print(f"\nResult: prompt={prompt} t/s | generation={gen} t/s | VRAM delta={vram_delta} MB")

    return result


# ─────────────────────────────────────────────
# BENCHMARK SUITE
# The set of experiments we run.
# Each config tests a specific variable.
# ─────────────────────────────────────────────

BENCHMARK_CONFIGS = [
    {
        "label": "full_gpu_ngl33",
        "ngl": 33,
        "n_tokens": 200,
        "prompt": "The theory of relativity states that",
        "temp": 0,
    },
    {
        "label": "half_gpu_ngl16",
        "ngl": 16,
        "n_tokens": 200,
        "prompt": "The theory of relativity states that",
        "temp": 0,
    },
    {
        "label": "cpu_only_ngl0",
        "ngl": 0,
        "n_tokens": 200,
        "prompt": "The theory of relativity states that",
        "temp": 0,
    },
    {
        "label": "full_gpu_long_context",
        "ngl": 33,
        "n_tokens": 500,
        "prompt": "The theory of relativity states that",
        "temp": 0,
    },
]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="llama.cpp benchmark suite")
    parser.add_argument("--binary", required=True, help="Path to llama-cli binary")
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument("--output", required=True, help="Path to save JSON results")
    parser.add_argument("--configs", default="all", help="Comma-separated config labels, or 'all'")
    args = parser.parse_args()

    # Validate paths
    if not Path(args.binary).exists():
        print(f"ERROR: binary not found: {args.binary}")
        return
    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}")
        return

    # Select configs to run
    if args.configs == "all":
        configs = BENCHMARK_CONFIGS
    else:
        labels = args.configs.split(",")
        configs = [c for c in BENCHMARK_CONFIGS if c["label"] in labels]

    print(f"Running {len(configs)} benchmark configurations")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    results = []
    for config in configs:
        result = run_single(args.binary, args.model, config)
        results.append(result)

        # Save after every run — don't lose data if something crashes
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {args.output}")

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"{'Config':<30} {'Prompt t/s':>12} {'Gen t/s':>12} {'VRAM delta':>12}")
    print("-" * 70)
    for r in results:
        label = r["label"]
        prompt = r["timing"].get("prompt_tokens_per_sec") or "N/A"
        gen = r["timing"].get("generation_tokens_per_sec") or "N/A"
        vram = r["memory"].get("vram_delta_mb") or "N/A"
        print(f"{label:<30} {str(prompt):>12} {str(gen):>12} {str(vram):>12}")


if __name__ == "__main__":
    main()