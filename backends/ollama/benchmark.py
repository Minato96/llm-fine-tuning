import requests
import time
import json

def run_single(model, prompt):
    start = time.time()
    latency = None
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":model,
            "prompt":prompt,
            "stream":True
        },
        stream=True
    )
    tokens_count = 0
    output=""
    for line in response.iter_lines():
        if line:
            line =json.loads(line)
            if latency is None and line['response']!="":
                latency = time.time()-start
            if line['response']!="":
                tokens_count += 1
            output += line['response']
            if(line['done']):
                elapsed = time.time() - start
                break

    return{
        "latency": latency,
        "elapsed":elapsed,
        "tokens_count": tokens_count,
        "throughput": tokens_count/(elapsed-latency),
        "output": output
    }

if __name__ == "__main__":
    result = run_single("phi3-local", "What is the meaning of life?")
    print(result)