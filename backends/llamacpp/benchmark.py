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
  python3 benchmark.py \
    --binary ~/projects/llm-fine-tuning/llama.cpp/build/bin/llama-cli \
    --model ~/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
    --output ~/projects/llm-fine-tuning/benchmarks/results/llamacpp.json
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
    WHY we use `script` here:
      llama.cpp writes its timing line [ Prompt: X t/s | Generation: Y t/s ]
      directly to /dev/tty (the physical terminal), bypassing stdout and stderr.
      Normal subprocess capture misses it entirely.

      `script` creates a pseudo-terminal (PTY) — a fake terminal that the program
      thinks is a real screen. Everything written to /dev/tty goes into the PTY,
      which `script` saves to a file. We then read that file to get ALL output
      including the timing line.

    WHY we strip ANSI codes:
      Terminal output contains escape sequences like \x1b[0m (color codes) and
      \x1b[2K (cursor controls). These are instructions to the terminal renderer,
      not actual text. If we don't strip them, our regex won't find the timing line
      because it's buried in escape code noise.
    """
    # Build the inner llama.cpp command as a shell string
    # (script -c takes a single string, not a list)
    inner_cmd = (
        f"{binary}"
        f" -m {model}"
        f" -n {config['n_tokens']}"
        f" -ngl {config['ngl']}"
        f" -p \"{config['prompt']}\""
        f" --temp {config.get('temp', 0)}"
        f" --single-turn"
    )

    # Temp file where script saves all terminal output
    capture_file = f"/tmp/llama_bench_{config['label']}.txt"

    # Outer command: script wraps the inner command
    # -q = quiet (no "Script started/stopped" header)
    # -c = command to run inside the fake terminal
    # last arg = file to save output to
    cmd = ["script", "-q", "-c", inner_cmd, capture_file]

    print(f"\n{'='*60}")
    print(f"Running: {config['label']}")
    print(f"  GPU layers : {config['ngl']}")
    print(f"  Tokens     : {config['n_tokens']}")
    print(f"{'='*60}")

    # Start GPU sampling BEFORE launching — we want to capture the model load spike
    sampler = GPUSampler(interval=0.5)
    sampler.start()

    start_time = time.time()
    subprocess.run(cmd)  # runs and blocks until llama.cpp exits
    elapsed = time.time() - start_time

    sampler.stop()

    # Read what script captured
    try:
        raw = Path(capture_file).read_text(errors="replace")
    except FileNotFoundError:
        raw = ""

    # Strip ANSI escape codes so our regex can find clean text
    # \x1b[ starts an escape sequence, [0-9;]* matches params, [a-zA-Z] ends it
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    clean = ansi_escape.sub('', raw)

    # Parse timing and memory from cleaned output
    parsed = parse_llamacpp_output(clean)

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
        "raw_output": clean
    }

    gen = parsed.get("generation_tokens_per_sec", "N/A")
    prompt_tps = parsed.get("prompt_tokens_per_sec", "N/A")
    vram_delta = sampler.summary().get("vram_delta_mb", "N/A")
    print(f"\nResult: prompt={prompt_tps} t/s | generation={gen} t/s | VRAM delta={vram_delta} MB")

    return result


# ─────────────────────────────────────────────
# BENCHMARK SUITE
# The set of experiments we run.
# Each config tests one specific variable.
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
    parser.add_argument("--model",  required=True, help="Path to .gguf model file")
    parser.add_argument("--output", required=True, help="Path to save JSON results")
    parser.add_argument("--configs", default="all",
                        help="Comma-separated config labels to run, or 'all'")
    args = parser.parse_args()

    if not Path(args.binary).exists():
        print(f"ERROR: binary not found: {args.binary}"); return
    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}"); return

    configs = BENCHMARK_CONFIGS if args.configs == "all" else [
        c for c in BENCHMARK_CONFIGS if c["label"] in args.configs.split(",")
    ]

    print(f"Running {len(configs)} benchmark configurations")
    print(f"Model : {args.model}")
    print(f"Output: {args.output}")

    results = []
    for config in configs:
        result = run_single(args.binary, args.model, config)
        results.append(result)

        # Save after every run so we don't lose data if something crashes
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} result(s) → {args.output}")

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'Config':<30} {'Prompt t/s':>12} {'Gen t/s':>12} {'VRAM delta':>12}")
    print("-" * 70)
    for r in results:
        p   = r["timing"].get("prompt_tokens_per_sec") or "N/A"
        g   = r["timing"].get("generation_tokens_per_sec") or "N/A"
        v   = r["memory"].get("vram_delta_mb") or "N/A"
        print(f"{r['label']:<30} {str(p):>12} {str(g):>12} {str(v):>12}")


if __name__ == "__main__":
    main()