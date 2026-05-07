import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define the model
model_id = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading tokenizer and model: {model_id}...")

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load Model (fp16 precision, directly to GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda" 
)
print("Model loaded successfully into VRAM!\n")

# 4. The SQL Prompt
prompt = "You are a database engineer. Write a PostgreSQL query to find the top 5 customers by total spend in the year 2023. Assume tables 'customers' and 'orders' (which has customer_id, order_date, and total_amount)."

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": prompt}
]

# Format the prompt using the model's specific chat template
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cuda")

# Reset PyTorch's VRAM tracker to get an accurate reading of the generation phase
torch.cuda.reset_peak_memory_stats()

print("Starting generation...")
start_time = time.time()

# 5. Generate the Output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1, # Low temperature for logical tasks like SQL
        do_sample=True
    )

end_time = time.time()

# 6. Calculate Metrics
# Strip the prompt from the output to only count generated tokens
generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

num_tokens = len(generated_tokens)
total_time = end_time - start_time
tokens_per_sec = num_tokens / total_time
# Convert bytes to Gigabytes
peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) 

print("\n--- Model Output ---")
print(response)
print("\n--- Baseline Benchmark Results ---")
print(f"Tokens Generated: {num_tokens}")
print(f"Generation Time:  {total_time:.2f} seconds")
print(f"Throughput:       {tokens_per_sec:.2f} tokens/sec")
print(f"Peak VRAM Usage:  {peak_vram_gb:.2f} GB")