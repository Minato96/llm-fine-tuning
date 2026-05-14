import time
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1",api_key="vllm")
def run_single(model, prompt):
    start = time.time() 
    tokens_count = 0
    output = ""
    latency = None
    response =client.completions.create(
        model=model,
        prompt=prompt,
        stream=True,
        max_tokens=200,
        temperature=0
    )
    for line in response:
        output+= line.choices[0].text
        if latency is None and line.choices[0].text!="":
            latency = time.time()-start
        if line.choices[0].text!="":
            tokens_count+=1
    elapsed = time.time() - start
    return  {
        "latency": latency,
        "elapsed":elapsed,
        "tokens_count": tokens_count,
        "throughput": tokens_count/(elapsed-latency),
        "output": output
    }

model="/home/charan/models/phi3-mini/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
prompt="what is general theory of relativity?"
print(run_single(model, prompt))
