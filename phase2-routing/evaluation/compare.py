import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from router.heuristic import classify_heuristic
from router.embedding import EmbeddingRouter
import time

def evaluate_router(router, test_prompts):
    start=time.time()
    correct = 0
   
        
    for prompt, expected in test_prompts:
        if router == classify_heuristic:
             result = router(prompt)
        else:
            result = router.route(prompt)
        got = result["model"]
        status = "✓" if got == expected else "✗"
        print(f"{status} [{expected}→{got}] {prompt[:50]}")
        if got == expected:
            correct += 1 
    end = time.time()
    print(f"\nEvaluation completed in {end - start:.2f} seconds.")
    print(f"\nAccuracy: {correct}/{len(test_prompts)} = {correct*100/len(test_prompts):.2f}%")

if __name__ == "__main__":
    test_prompts = [
    # original 10
    ("hi how are you?", "qwen2.5:0.5b"),
    ("what is the capital of france?", "qwen2.5:0.5b"),
    ("calculate 2+2", "qwen2.5:0.5b"),
    ("how many days are there in a week", "qwen2.5:0.5b"),
    ("what does hola mean in english", "qwen2.5:0.5b"),
    ("prove the riemann hypothesis", "phi3-local"),
    ("explain quantum entanglement", "phi3-local"),
    ("why is the universe left oriented and how is it proved", "phi3-local"),
    ("why does switching llms not cause more latency", "phi3-local"),
    ("explain how time is born from the big bang", "phi3-local"),
    # 5 new ones your routers haven't seen
    ("implement a binary search tree in python", "phi3-local"),
    ("what causes inflation", "phi3-local"),
    ("who wrote hamlet", "qwen2.5:0.5b"),
    ("what is the boiling point of water", "qwen2.5:0.5b"),
    ("analyze the ethical implications of autonomous weapons", "phi3-local"),
]

    print("Evaluating Heuristic Router:")
    evaluate_router(classify_heuristic, test_prompts)

    print("\nEvaluating Embedding Router:")
    embedding_router = EmbeddingRouter()
    evaluate_router(embedding_router, test_prompts)
