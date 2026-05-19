import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI, routing
from pydantic import BaseModel
from phase2_routing.router.heuristic import classify_heuristic
from phase2_routing.router.embedding import EmbeddingRouter
from backends.ollama.benchmark import run_single

app = FastAPI()
embedding_router = EmbeddingRouter()
@app.get("/test")
def test():
    return("test successful")

@app.post("/route")
def route_query(request: dict):
    query = request.get("query")
    if not query:
        return {"error": "Query is required."}
    
    embed_type = request.get("embed_type", "embedding")  # default to embedding if not specified
    # Heuristic routing
    if embed_type == "heuristic":
        heuristic_result = classify_heuristic(query)
        result = run_single(heuristic_result["model"], query)
        return {
            "routing_decision": heuristic_result,
            "model_response": result
        }
    # Embedding routing
    elif embed_type == "embedding":
        embedding_result = embedding_router.route(query)
        result = run_single(embedding_result["model"], query)
        return {
            "routing_decision": embedding_result,
            "model_response": result
        }
    

