# LLM Inference & Serving Platform

A local LLM inference benchmarking and serving platform built on consumer hardware — RTX 5060 8GB VRAM. This project explores the full stack of production LLM serving: from bare-metal GPU inference to intelligent routing to fine-tuning, measuring every tradeoff with real telemetry.

**Hardware:** NVIDIA RTX 5060 Laptop GPU · 8GB VRAM · Blackwell sm_120 · CUDA 12.8  
**Model:** Phi-3-mini-4k-instruct Q4_K_M (3.8B params, 2.4GB on disk)  
**Stack:** Python · llama.cpp · Ollama · vLLM · FastAPI · Chart.js

---

## Motivation

Most AI engineering tutorials teach you to call an API. This project goes underneath — understanding why one inference setup is 10x faster than another, how memory fills up, what paged attention actually solves, and how routing decisions affect system behavior under load.

The constraint of 8GB VRAM isn't a limitation. It forces genuine understanding of memory math, quantization tradeoffs, and optimization decisions that cloud users never encounter.

---

## Project Structure

```
llm-fine-tuning/
├── backends/
│   ├── llamacpp/benchmark.py     # bare-metal GPU inference benchmarks
│   ├── ollama/benchmark.py       # HTTP streaming client + TTFT measurement
│   └── vllm/benchmark.py         # OpenAI-compatible client + concurrency
├── benchmarks/
│   ├── runner.py                 # concurrent load testing across backends
│   └── results/
│       ├── llamacpp_phi3.json    # GPU layer split benchmark results
│       ├── ollama_concurrent.json
│       └── vllm_concurrent.json
├── monitoring/
│   └── gpu_watch.py              # real-time VRAM + GPU utilization sampler
├── dashboard/
│   └── dashboard.html            # interactive benchmark visualization
├── phase2-routing/               # intelligent prompt routing (Phase 2)
│   ├── router/
│   │   ├── heuristic.py
│   │   └── embedding.py
│   ├── gateway/app.py
│   └── evaluation/test_prompts.json
└── llama.cpp/                    # built from source with CUDA sm_120
```

---

## Phase 1 — High Performance Inference Core

### What was built

Three inference backends benchmarked against the same model and prompts:

- **llama.cpp** — built from source with CUDA sm_120 (Blackwell). Benchmarked GPU layer splits: full GPU (ngl=33), half GPU (ngl=16), CPU only (ngl=0).
- **Ollama** — HTTP serving layer over llama.cpp. Wrote streaming client from scratch measuring TTFT and generation throughput.
- **vLLM** — production inference engine with paged attention and continuous batching. Tested under concurrent load with threading.

### Key findings

**1. LLM generation is memory-bandwidth-bound, not compute-bound**

For a single request, the GPU loads ~2,350MB of weights to produce each token. At 272 GB/s VRAM bandwidth, the theoretical ceiling is ~116 tokens/sec. Measured generation speed of 83.5 tokens/sec (72% efficiency) confirms the bottleneck is memory bandwidth, not arithmetic throughput.

During prefill (processing the input prompt), all tokens are processed as a matrix-matrix multiply — high arithmetic intensity, GPU hits 92% utilization and 71W power draw. During generation (one token at a time), arithmetic intensity drops to ~3.5 FLOPs/byte, GPU utilization falls to 10-33%.

**2. The PCIe split penalty is severe**

Splitting layers between GPU and CPU (ngl=16) produced 14.0 tokens/sec — barely better than pure CPU at 8.4 tokens/sec, despite half the model being on the GPU.

The reason: every token requires data to cross the PCIe bus twice (GPU→CPU and CPU→GPU). At ~75μs per transfer × 200 tokens, the synchronization overhead dominates. The practical implication: either fit the model fully in VRAM, or run pure CPU. The hybrid middle ground is almost never worth it.

```
Config          Prompt t/s    Gen t/s    VRAM Delta
Full GPU            165.5       83.5       3,957 MB
Half GPU (ngl=16)    58.3       14.0       2,015 MB   ← PCIe bottleneck
CPU only              47.1        8.4         247 MB
```

**3. Single-request performance is similar across backends — for a reason**

llama.cpp, Ollama, and vLLM all produce ~83-91 tokens/sec for a single request on the same model. This is not a coincidence — all three are limited by the same VRAM memory bandwidth ceiling. Architecture differences only manifest under concurrent load.

**4. vLLM's continuous batching vs Ollama's sequential serving**

Under concurrent load, vLLM and Ollama diverge dramatically:

```
Backend    n=1       n=2       n=4       TTFT @ n=4
vLLM       71.7      124.9     190.1     0.10s
Ollama     88.0      112.6      91.3     10.04s
```

vLLM at n=4 achieves 190 tokens/sec — above the single-request ceiling — because continuous batching stacks multiple requests into one GPU forward pass. The effective bytes-per-token drops from 2,350MB to ~587MB, quadrupling throughput.

Ollama's total throughput stays flat at ~91 tokens/sec regardless of concurrency because requests are served sequentially. The fourth user waits for the first three to finish, producing a 10-second TTFT — a real product failure.

**5. Paged attention — the KV cache problem vLLM solves**

Before vLLM, KV cache was pre-allocated as a contiguous block for the maximum sequence length. A request generating 50 tokens reserved space for 2048 tokens — 97.5% waste. Under concurrent load, VRAM appeared full while mostly sitting empty.

vLLM applies OS virtual memory concepts to KV cache: dividing VRAM into fixed-size KV blocks (~16 tokens each) allocated on demand. A 50-token request gets 4 blocks. A 500-token request gets 32 blocks. No fragmentation. vLLM calculated 10,496 tokens of KV cache capacity on this hardware, supporting 5.12 concurrent requests at max context length.

### Quantization

The model is stored in Q4_K_M format — an average of 4.5 bits per weight versus 16 bits in float16. This reduces the 14GB float16 model to 2.4GB, making it fit in 8GB VRAM with room for KV cache. The "K" indicates K-quants (mixed precision, keeping sensitive layers at higher precision), "M" is medium quality-size balance.

Quantization matters beyond just fitting in VRAM — it also increases inference speed because less data needs to be loaded from VRAM per token.

### Running the benchmarks

```bash
# Install dependencies
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.20.2 requests openai huggingface-hub

# Build llama.cpp with CUDA
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j$(nproc)

# Download model
hf download bartowski/Phi-3-mini-4k-instruct-GGUF \
  --include "Phi-3-mini-4k-instruct-Q4_K_M.gguf" \
  --local-dir ~/models/phi3-mini

# Run llama.cpp benchmark
python backends/llamacpp/benchmark.py \
  --binary llama.cpp/build/bin/llama-cli \
  --model ~/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --output benchmarks/results/llamacpp_phi3.json

# Start Ollama and run benchmark
ollama create phi3-local -f /tmp/Modelfile
python backends/ollama/benchmark.py

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --tokenizer microsoft/Phi-3-mini-4k-instruct \
  --quantization gguf --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 --port 8000

# Run concurrent load test
python benchmarks/runner.py

# Open dashboard
open dashboard/dashboard.html
```

### GPU monitoring

```bash
# Real-time VRAM and utilization
python monitoring/gpu_watch.py

# Quick snapshot
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv
```

---

## Phase 2 — Intelligent Routing Layer

*(In progress)*

A routing engine that classifies incoming prompts and directs them to the appropriate model:

- Simple queries (factual, single-answer) → `qwen2.5:0.5b` (400MB, fast)
- Complex queries (reasoning, generation) → `phi3-mini` (2.4GB, higher quality)

Both models fit simultaneously in 8GB VRAM (combined ~3.3GB weights + KV cache). Zero model swapping cost.

Two routing strategies under development:
1. **Heuristic router** — keyword-based classification, <1ms latency
2. **Embedding router** — sentence similarity to labeled examples, ~15ms latency

---

## Phases 3-8

| Phase | Topic | Status |
|-------|-------|--------|
| 3 | LoRA fine-tuning + Unsloth + full model lifecycle | Planned |
| 4 | Adaptive memory architecture | Planned |
| 5 | Retrieval Engine 2.0 (hybrid BM25 + vector) | Planned |
| 6 | Multi-agent execution layer | Planned |
| 7 | Evaluation infrastructure | Planned |
| 8 | Production systems (Redis, Prometheus, Grafana) | Planned |

---

## Hardware Baseline

```
GPU:          NVIDIA GeForce RTX 5060 Laptop GPU
VRAM:         8,151 MiB total / 7,994 MiB usable
Bandwidth:    ~272 GB/s
Architecture: Blackwell (sm_120)
Driver:       596.36 (Windows) / 595.71.01 (WSL2)
CUDA:         12.8.93
Idle VRAM:    ~157 MiB (1.9%)
Idle Power:   ~3.5W / 80W TDP
Idle Temp:    42°C
```

---

*Built on WSL2 Ubuntu 24.04 · Python 3.12 · llama.cpp b9093*