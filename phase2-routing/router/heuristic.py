def classify_heuristic(prompt: str) -> dict:
    reason = []
    score = 0
    prompt = prompt.lower()
    if "explain" in prompt or "prove" in prompt or "derive" in prompt or "compare" in prompt or "analyze" in prompt or "why" in prompt:
        reason.append("Contains complexity-indicating keywords")
        score += 6
    
    if len(prompt.split()) > 20:
        reason.append("Prompt is long")
        score += 4
    
    if "write" in prompt or "generate" in prompt or "create" in prompt:
        reason.append("Contains generation-indicating keywords")
        score += 5
    
    if score>=5:
        return{
            "model":"phi3-local",
            "reason": reason,
            "score": score,
            "complexity": "high"
        }
    else:
        return{
            "model": "qwen2.5:0.5b",
            "reason": reason,
            "score": score,
            "complexity": "low"
        }

if __name__ == "__main__":
    test_prompts = [
        ("hi how are you?", "low"),
        ("what is the capital of france?", "low"),
        ("calculate 2+2", "low"),
        ("how many days are there in a week", "low"),
        ("what does hola mean in english", "low"),
        ("prove the riemann hypothesis", "high"),
        ("explain quantum entanglement", "high"),
        ("why is the universe left oriented and how is it proved", "high"),
        ("why does switching llms not cause more latency", "high"),
        ("explain how time is born from the big bang", "high"),
    ]

    correct = 0
    for prompt, expected in test_prompts:
        result = classify_heuristic(prompt)
        got = result["complexity"]
        status = "✓" if got == expected else "✗"
        print(f"{status} [{expected}→{got}] {prompt[:50]}")
        if got == expected:
            correct += 1

    print(f"\nAccuracy: {correct}/{len(test_prompts)} = {correct*100/len(test_prompts):.2f}%")