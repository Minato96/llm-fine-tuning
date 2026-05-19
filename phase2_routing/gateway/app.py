import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI, routing
from pydantic import BaseModel
from phase2-routing.router.heuristic import classify_heuristic
from phase2-routing.router.embedding import EmbeddingRouter
from backends.ollama.benchmark import run_single

app = FastAPI()

@app.get("/test")
def test():
    return("test successful")

@app.post("/route")
def route_query(request: dict):
    query = request.get("query")
    if not query:
        return {"error": "Query is required."}
    
    # Heuristic routing
    heuristic_result = classify_heuristic(query)
    
    # Embedding routing
    embedding_router = EmbeddingRouter()
    embedding_result = embedding_router.route(query)
    
    return {
        "heuristic": heuristic_result,
        "embedding": embedding_result
    }

