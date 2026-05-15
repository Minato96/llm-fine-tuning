import os
import sys

from openai import OpenAI

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
import json
from pathlib import Path
import threading
import time
from backends.ollama.benchmark import run_single as ollama_run_single
from backends.vllm.benchmark import run_single as vllm_run_single


client = OpenAI(base_url="http://localhost:8000/v1",api_key="vllm")
def worker(results,model, prompt, backend):
    if backend == "ollama":
        result = ollama_run_single(model, prompt)
    elif backend == "vllm":
        result = vllm_run_single(model, prompt,client)
    else:
        print(f"Unknown backend: {backend}")
        return
    results.append(result)

def run_concurrent(model,prompt,backend, num_threads):
    results = []
    threads = []
    for i in range(num_threads):
        t =threading.Thread(target=worker,args=(results,model,prompt,backend))
        threads.append(t)
    
    wall_start = time.time()
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    wall_end = time.time()
    wall_time = wall_end - wall_start
    total_tokens = sum(r['tokens_count'] for r in results)
    avg_latency = sum(r['latency'] for r in results) / num_threads
    throughput = total_tokens / wall_time
    return {
        "n_concurrent": num_threads,
        "wall_time_sec": round(wall_time, 2),
        "total_tokens": total_tokens,
        "total_throughput_tokens_per_sec": round(throughput, 1),
        "avg_latency_sec": round(avg_latency, 2),
        "individual_results": results
    }



vllm_results = []
client = OpenAI(base_url="http://localhost:8000/v1",api_key="vllm")
for n in [1, 2, 4]:
    out = run_concurrent(
        model="/home/charan/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        prompt="what is general theory of relativity?",
        backend="vllm",
        num_threads=n
    )  # vllm
    vllm_results.append(out)

# ollama_results = []
# for n in [1, 2, 4]:
#     out = run_concurrent(
#         model="phi3-local",
#         prompt="what is general theory of relativity?",
#         backend="ollama",
#         num_threads=n
#     )
#     ollama_results.append(out)

Path("benchmarks/results").mkdir(exist_ok=True)
json.dump(vllm_results, open("benchmarks/results/vllm_concurrent.json", "w"), indent=2)
# json.dump(ollama_results, open("benchmarks/results/ollama_concurrent.json", "w"), indent=2)