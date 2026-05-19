# Phase 1 — High Performance Inference Core

**Goal:** Understand how LLM serving actually works at the hardware level. Build benchmark infrastructure that produces real telemetry — not screenshots.

← [Back to README](README.md) | [Phase 2 →](phase2_routing/PHASE2.md)

---

## What Was Built

Three inference backends benchmarked against the same model, same prompts, same hardware:

- **llama.cpp** — C++ runtime built from source with CUDA sm_120. Direct hardware control, no abstraction. Benchmarked GPU layer splits.
- **Ollama** — HTTP serving layer over llama.cpp. Persistent model loading, REST API. HTTP streaming client written from scratch.
- **vLLM** — Production inference engine. Paged attention, continuous batching, OpenAI-compatible API. Benchmarked under concurrent load.

**Model:** Phi-3-mini-4k-instruct Q4_K_M across all three backends for fair comparison.

---

## Core Concept: Why LLM Inference Is Hard

The fundamental tension: a GPU can compute much faster than it can fetch data from VRAM.

```
RTX 5060 peak compute:    ~22 TFLOPS
RTX 5060 memory bandwidth: ~272 GB/s

Roofline threshold: 22,000 / 272 = ~80 FLOPs/byte
```

For single-token generation (matrix × vector):
```
Arithmetic intensity = 2 × d_model / bytes_per_weight
                     = 2 × 3072 / 0.5625
                     = 10.9 FLOPs/byte  ←  far below threshold
```

**The GPU is memory-bandwidth-bound during generation.** It's not waiting to do math — it's waiting for weight data to arrive from VRAM. This single insight explains every benchmark result in Phase 1.

---

## Quantization: Why It's Necessary

Phi-3-mini has 3.82 billion parameters. In float16 (2 bytes/param): **14GB**. Doesn't fit in 8GB VRAM.

Q4_K_M quantization stores weights at an average of 4.5 bits/param:
```
3.82 × 10⁹ × 4.5 bits / 8 = 2.15 GB weights
+ vocabulary embeddings + metadata ≈ 2.4 GB total
```

The "K" in Q4_K_M means K-quants — a mixed precision approach that keeps sensitive layers (early layers, attention projections) at higher precision while aggressively quantizing less sensitive ones. "M" is medium — a balance between size and quality.

Quantization matters for two reasons: (1) model fits in VRAM at all, (2) fewer bytes per token = faster generation due to lower memory bandwidth demand.

---

## Benchmark Results

### GPU Layer Split (llama.cpp)

```
Config          Prompt t/s    Gen t/s    VRAM Delta    Wall time (200 tokens)
Full GPU(ngl=33)   165.5        83.5       3,957 MB        4.74s
Half GPU(ngl=16)    58.3        14.0       2,015 MB       15.25s
CPU only(ngl=0)     47.1         8.4         247 MB       25.36s
```

**The PCIe penalty is severe.** Half GPU (ngl=16) gives 14.0 t/s — barely better than pure CPU at 8.4 t/s despite half the model being on the GPU.

Why: every token requires data to cross the PCIe bus twice (GPU→CPU then CPU→GPU). PCIe 4.0 x8 has ~75μs per transfer latency. Over 200 tokens: 200 × 75μs × 2 = 30ms just in transfer overhead, plus the CPU layers themselves are bandwidth-limited by RAM (~50 GB/s vs 272 GB/s VRAM).

**Memory math for full GPU:**
```
Model weights:   2,228 MB   (Q4_K_M weights in VRAM)
KV cache:        1,536 MB   (pre-allocated for 4K context)
Compute buffers:    74 MB
─────────────────────────────
Total:           3,838 MB   (matches measured 3,957 MB delta from baseline)
```

The KV cache pre-allocation is the key issue — even for a 200-token generation, llama.cpp reserves space for the full 4,096-token context window upfront. This fragmentation problem is what vLLM's paged attention solves.

### Theoretical vs Measured Generation Speed

```
Theoretical ceiling = VRAM bandwidth / bytes per token
                    = 272,000 MB/s / 2,350 MB
                    = 115.7 tokens/sec

Measured:             83.5 tokens/sec
Efficiency:           72.1%
```

Gap explained: dequantization cost (~15%), kernel launch overhead (~5%), KV cache reads (additional memory traffic not captured in weight-only calculation), CUDA synchronization between layers.

### Concurrent Load (Ollama vs vLLM)

```
Backend    n=1       n=2       n=4       TTFT@n=4    Wall@n=4
vLLM       71.7      124.9     190.1     0.10s        4.89s
Ollama     88.0      112.6      91.3     10.04s       26.6s
```

**Why vLLM throughput scales above the single-request ceiling:**

At n=4, vLLM batches all requests into one forward pass per step — matrix × matrix instead of matrix × vector. Effective bytes-per-token:
```
Single request: 2,350 MB / 1 token = 2,350 MB/token
Batch of 4:     2,350 MB / 4 tokens = 587 MB/token

New ceiling:    272,000 / 587 = ~464 tokens/sec
```

Measured 190 t/s is below the new ceiling due to KV cache overhead and request synchronization — but well above the single-request ceiling.

**Why Ollama's latency explodes:**

Ollama serves requests sequentially. At n=4, request 4 waits for requests 1, 2, 3 to fully complete before receiving its first token. Average TTFT at n=4: 10.04 seconds. In a real product, that user has already closed the tab.

### Paged Attention — The Architecture That Makes the Difference

Before vLLM, KV cache was pre-allocated as one contiguous block per request:
```
Max context = 2,048 tokens
KV cache per request = 2 × 32 layers × 32 heads × 96 d_head × 2048 × 2 bytes = 768 MB
Reserved even if request only generates 50 tokens → 96% waste
```

vLLM divides VRAM into fixed-size KV blocks (16 tokens each). A 50-token request gets 4 blocks. A 500-token request gets 32 blocks. Nothing pre-reserved. Same concept as OS virtual memory paging.

From the actual startup log:
```
Available KV cache memory: 3.85 GiB
GPU KV cache size: 10,496 tokens
Maximum concurrency for 2,048 tokens per request: 5.12x
```

vLLM knew its exact serving capacity before the first request arrived. llama.cpp has no concept of this.

---

## VRAM Timeline

Captured via `gpu_watch.py` sampling nvidia-smi every 500ms during a full GPU run:

```
Time    VRAM Used    Event
0.0s    91 MB        baseline
0.5s    192 MB       CUDA context initializing
1.0s    2,422 MB     model weights arriving via PCIe
1.5s    4,048 MB     KV cache allocated, inference starting
2.0s    4,048 MB     generating tokens
...
end     91 MB        process exited, VRAM released
```

The jump from 91MB to 4,048MB happens in two steps — weights load first (91→2,422MB), then KV cache is allocated (2,422→4,048MB). These are visually distinct in the dashboard.

---

## Key Insight: Single Request Performance Is Similar Across All Backends

llama.cpp, Ollama, and vLLM all produce 71-91 tokens/sec for a single request. This is not a coincidence — all three are bounded by the same VRAM memory bandwidth ceiling. Architecture differences only manifest under concurrent load. This was an empirically verified finding, not an assumption.

---

## Running Phase 1

```bash
# Build llama.cpp with CUDA (Blackwell sm_120)
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j$(nproc)

# Download model
hf download bartowski/Phi-3-mini-4k-instruct-GGUF \
  --include "Phi-3-mini-4k-instruct-Q4_K_M.gguf" \
  --local-dir ~/models/phi3-mini

# Run llama.cpp benchmarks
python backends/llamacpp/benchmark.py \
  --binary llama.cpp/build/bin/llama-cli \
  --model ~/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --output benchmarks/results/llamacpp_phi3.json

# Import model to Ollama (reuses existing GGUF, no re-download)
echo "FROM /home/charan/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf" > /tmp/Modelfile
ollama create phi3-local -f /tmp/Modelfile

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --tokenizer microsoft/Phi-3-mini-4k-instruct \
  --quantization gguf --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 --port 8000

# Run concurrent load test (vLLM must be running, Ollama must be running)
python benchmarks/runner.py

# View dashboard
open dashboard/app.html

# Monitor GPU in real time
python monitoring/gpu_watch.py
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `backends/llamacpp/benchmark.py` | Runs llama-cli via `script` PTY capture, parses timing + memory breakdown, GPUSampler thread |
| `backends/ollama/benchmark.py` | HTTP streaming client, measures TTFT and generation throughput |
| `backends/vllm/benchmark.py` | OpenAI SDK client, streaming, measures TTFT |
| `benchmarks/runner.py` | Threading-based concurrent load test, saves JSON results |
| `monitoring/gpu_watch.py` | nvidia-smi polling loop, saves time-series JSON |
| `dashboard/app.html` | Chart.js visualization of all benchmark results |

---

← [Back to README](README.md) | [Phase 2 →](phase2_routing/PHASE2.md)