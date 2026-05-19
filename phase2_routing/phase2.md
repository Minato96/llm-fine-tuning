# Phase 2 — Intelligent Routing Layer

**Goal:** Build a routing engine that analyzes incoming prompts and directs them to the appropriate model — balancing quality, speed, and resource usage.

← [Phase 1](../PHASE1.md) | [Back to README](../README.md) | Phase 3 →

---

## The Problem

After Phase 1, every request goes to the same model (phi3-mini). This is wasteful:

- "What is the capital of France?" doesn't need a 3.8B parameter model
- "Explain quantum entanglement" absolutely does
- A 0.5B model answers factual questions correctly and returns in 150ms
- The same 0.5B model gives confidently wrong answers on complex reasoning

The routing layer solves this: classify the prompt first, then choose the right model. Simple → fast tiny model. Complex → capable large model.

---

## Hardware Reality Check

Both models fit simultaneously in 8GB VRAM with zero swapping:

```
phi3-local (3.8B Q4_K_M):    ~2,228 MB weights + ~1,536 MB KV cache
qwen2.5:0.5b:                ~400 MB weights + ~400 MB KV cache
CUDA overhead:               ~300 MB
─────────────────────────────────────────────────
Total:                       ~4,864 MB  (of 8,151 MB available)
Headroom:                    ~3,287 MB free
```

Confirmed by nvidia-smi after loading both:
```
$ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
5258
```

No model swapping needed. The router's only job is to pick which already-loaded model handles the request.

---

## Two Routing Strategies

### Strategy 1 — Heuristic Router

Rule-based classification using keyword matching and prompt length. Written entirely from scratch — no ML model, no embeddings. Decision in under 1ms.

**How it works:**

Assigns a complexity score to the prompt based on signals:

```python
if "explain" or "prove" or "derive" or "compare" or "analyze" or "why" in prompt:
    score += 6   # strong complexity signal

if word_count > 20:
    score += 4   # length signal

if "write" or "generate" or "create" in prompt:
    score += 5   # generation task signal

complexity = "high" if score >= 5 else "low"
model = "phi3-local" if high else "qwen2.5:0.5b"
```

Returns not just the routing decision but the reason — fully explainable.

**Why score-based instead of pure keyword matching:**

A prompt that's long AND has a complexity keyword gets score 10. A prompt that's just long gets score 4 (routes to simple model). Weighted scoring handles ambiguous cases that binary keyword matching can't.

**Known failure modes:**

- "implement a binary search tree" → no complexity keywords → score 0 → wrongly routes to qwen2.5
- "what causes inflation" → no keywords, short → score 0 → wrongly routes to qwen2.5

These are prompts that are complex by task type but don't contain explicit complexity signal words. Heuristic routers are fundamentally blind to task type — they only see surface-level patterns.

### Strategy 2 — Embedding Router

Semantic classification using sentence embeddings and k-Nearest Neighbors. Generalizes to prompt types not covered by explicit rules. Decision in ~10ms.

**How it works:**

At startup, embeds all labeled example prompts using `sentence-transformers/all-MiniLM-L6-v2` (22MB model, runs in ~5ms). Fits a kNN classifier (k=3, cosine distance) on those embeddings.

At inference:
1. Embed the incoming prompt (~5ms)
2. Find the 3 most similar labeled examples
3. Majority vote on their labels → routing decision

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn.fit(labeled_embeddings, labels)  # at startup

# at inference:
embedding = model.encode(prompt).reshape(1, -1)
predicted_model = knn.predict(embedding)[0]
```

**Why kNN over averaging similarity:**

Instead of averaging similarity to all examples in each group, kNN finds the K most similar examples regardless of group. If "implement a binary search tree" is most similar to other coding examples labeled phi3, it correctly routes there — even without the word "implement" in the keyword list.

**Known failure modes:**

- "who wrote hamlet" → similar in embedding space to literary analysis prompts → wrongly routes to phi3
- Out-of-distribution prompts (domain not covered by training examples) → uncertain predictions

The embedding router fails *confidently* on out-of-distribution examples. The heuristic router fails *obviously* (missing keywords). Confident wrong answers are harder to debug than obvious ones.

---

## Head-to-Head Evaluation

Tested on 15 prompts: 10 original test cases + 5 new unseen prompts.

```
Router       Accuracy    Avg Latency    Failure Cases
Heuristic    86.67%      <1ms           implement BST, what causes inflation
Embedding    93.33%      ~10ms          who wrote hamlet
```

**The tradeoff in plain terms:**

- Embedding is 6.66% more accurate on this test set
- Embedding adds ~10ms latency per request
- Heuristic is fully explainable — you can log exactly why each decision was made
- Embedding generalizes better to unseen prompt types
- Both fail differently — heuristic on missing keywords, embedding on wrong cluster membership

**For production use:**

Use heuristic when: latency is critical, prompt distribution is predictable, explainability is required (regulated industries).

Use embedding when: prompt distribution is diverse and unpredictable, 10ms overhead is acceptable, you can maintain a labeled example dataset.

---

## FastAPI Gateway

The gateway ties everything together. A running HTTP server that accepts prompts, routes them, and returns the model's response.

**Endpoint:** `POST /route`

**Request:**
```json
{
  "query": "explain quantum entanglement",
  "embed_type": "heuristic"
}
```

**Response:**
```json
{
  "routing_decision": {
    "model": "phi3-local",
    "reason": ["Contains complexity-indicating keywords"],
    "score": 6,
    "complexity": "high"
  },
  "model_response": {
    "latency": 0.21,
    "elapsed": 3.90,
    "tokens_count": 351,
    "throughput": 95.16,
    "output": "Quantum entanglement is..."
  }
}
```

**Key design decision — EmbeddingRouter initialized at startup, not per-request:**

```python
# Module level — runs once when server starts
embedding_router = EmbeddingRouter()   # loads 22MB model, pre-computes embeddings

@app.post("/route")
def route_query(request: dict):
    result = embedding_router.route(request["query"])  # ~10ms, no model loading
```

If initialized inside the endpoint function, every request would reload the 22MB sentence transformer model — adding 5-10 seconds of latency. Startup cost paid once, amortized across all requests. Same principle as Ollama keeping models in VRAM.

**Observed latency (warm models):**

```
Simple query → qwen2.5:0.5b:   TTFT 0.15s, total 0.73s (149 tokens)
Complex query → phi3-local:    TTFT 0.21s, total 3.90s (351 tokens)
```

Higher total time for complex queries is expected and correct — phi3 generates more thorough responses, more tokens.

---

## Running Phase 2

```bash
# Pull routing models
ollama pull qwen2.5:0.5b
# phi3-local should already exist from Phase 1

# Start gateway
uvicorn phase2_routing.gateway.app:app --port 8080 --reload

# Test heuristic routing
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the capital of france?", "embed_type": "heuristic"}'

# Test embedding routing
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{"query": "explain quantum entanglement", "embed_type": "embedding"}'

# Run router comparison evaluation
python phase2_routing/evaluation/compare.py
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `router/heuristic.py` | Keyword + score based classifier, returns model + reason + score |
| `router/embedding.py` | EmbeddingRouter class: loads sentence-transformer, fits kNN, classifies |
| `gateway/app.py` | FastAPI server, routes requests to appropriate Ollama model |
| `evaluation/compare.py` | Side-by-side accuracy + latency evaluation of both routers |
| `evaluation/train_prompts.json` | Labeled prompt dataset (10 simple, 10 complex across diverse domains) |

---

← [Phase 1](../PHASE1.md) | [Back to README](../README.md) | Phase 3 →