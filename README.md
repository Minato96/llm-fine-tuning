# LLM Inference & Serving Platform

A local LLM inference, routing, and fine-tuning platform built entirely on consumer hardware — RTX 5060 8GB VRAM. This project explores the full production AI engineering stack: from bare-metal GPU inference to intelligent prompt routing to LoRA fine-tuning, with real telemetry at every layer.

**Hardware:** NVIDIA RTX 5060 Laptop GPU · 8GB VRAM · Blackwell sm_120 · CUDA 12.8  
**Primary Model:** Phi-3-mini-4k-instruct Q4_K_M (3.8B params, 2.4GB on disk)  
**Stack:** Python · llama.cpp · Ollama · vLLM · FastAPI · sentence-transformers · Unsloth · Chart.js

---

## Why This Project Exists

Most AI engineering tutorials teach you to call an API. This project goes underneath — understanding why one inference setup is 10x faster than another, how memory fills up under concurrent load, what paged attention actually solves, when to route a prompt to a smaller model, and how LoRA makes fine-tuning possible on 8GB VRAM.

The constraint of consumer hardware isn't a limitation. It forces genuine understanding of memory math, quantization tradeoffs, and optimization decisions that cloud users never encounter. Every number in this project was measured on real hardware, not simulated.

---

## Project Map

| Phase | Topic | Status | Details |
|-------|-------|--------|---------|
| 1 | High Performance Inference Core | ✅ Complete | [PHASE1.md](PHASE1.md) |
| 2 | Intelligent Routing Layer | ✅ Complete | [PHASE2.md](phase2_routing/PHASE2.md) |
| 3 | LoRA Fine-tuning + Full Model Lifecycle | 🔄 In Progress | — |
| 4 | Adaptive Memory Architecture | ⬜ Planned | — |
| 5 | Retrieval Engine 2.0 | ⬜ Planned | — |
| 6 | Multi-Agent Execution Layer | ⬜ Planned | — |
| 7 | Evaluation Infrastructure | ⬜ Planned | — |
| 8 | Production Systems Engineering | ⬜ Planned | — |

---

## Repository Structure

```
llm-fine-tuning/
├── README.md                          ← you are here
├── PHASE1.md                          ← Phase 1 deep dive
│
├── backends/                          ← inference backend benchmarks
│   ├── llamacpp/benchmark.py          # bare-metal GPU inference, GPU layer splits
│   ├── ollama/benchmark.py            # HTTP streaming client, TTFT measurement
│   └── vllm/benchmark.py             # OpenAI-compatible client, concurrency test
│
├── benchmarks/
│   ├── runner.py                      # concurrent load testing across backends
│   └── results/
│       ├── llamacpp_phi3.json         # GPU layer split results
│       ├── ollama_concurrent.json     # Ollama under concurrent load
│       └── vllm_concurrent.json      # vLLM under concurrent load
│
├── monitoring/
│   └── gpu_watch.py                   # real-time VRAM + utilization sampler
│
├── dashboard/
│   └── app.html                       # interactive benchmark dashboard
│
├── phase2_routing/                    ← Phase 2: intelligent routing
│   ├── PHASE2.md                      # Phase 2 deep dive
│   ├── router/
│   │   ├── heuristic.py               # keyword + score based router (<1ms)
│   │   └── embedding.py              # kNN + sentence embeddings router (~10ms)
│   ├── gateway/
│   │   └── app.py                     # FastAPI gateway (port 8080)
│   ├── evaluation/
│   │   ├── compare.py                 # head-to-head router accuracy test
│   │   └── train_prompts.json         # labeled prompt dataset
│   └── results/                       # routing evaluation outputs
│
└── llama.cpp/                         # built from source, CUDA sm_120
```

---

## Hardware Baseline

```
GPU:          NVIDIA GeForce RTX 5060 Laptop GPU
VRAM:         8,151 MiB total
Bandwidth:    ~272 GB/s
Architecture: Blackwell (sm_120)
Driver:       596.36 (Windows) / 595.71.01 (WSL2)
CUDA:         12.8.93
Idle VRAM:    ~157 MiB (1.9%)
Idle Power:   ~3.5W / 80W TDP
Idle Temp:    42°C
Platform:     WSL2 Ubuntu 24.04 · Python 3.12
```

---

## Quick Start

```bash
# Clone and set up environment
git clone <repo>
cd llm-fine-tuning
python -m venv venv && source venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install vllm requests openai huggingface-hub sentence-transformers fastapi uvicorn

# Download model
hf download bartowski/Phi-3-mini-4k-instruct-GGUF \
  --include "Phi-3-mini-4k-instruct-Q4_K_M.gguf" \
  --local-dir ~/models/phi3-mini

# Pull routing models
ollama pull qwen2.5:0.5b
ollama create phi3-local -f /tmp/Modelfile  # see PHASE1.md

# Start the routing gateway
uvicorn phase2_routing.gateway.app:app --port 8080

# Send a request
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{"query": "explain quantum entanglement", "embed_type": "heuristic"}'
```

---

*See individual phase docs for deep technical details, benchmark methodology, and key findings.*