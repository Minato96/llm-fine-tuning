from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
import json
import time

class EmbeddingRouter:
    def __init__(self,weak_model="qwen2.5:0.5b", strong_model="phi3-local", n_neighbors=3):
        self.weak_model = weak_model
        self.strong_model = strong_model
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors,metric='cosine')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.routes = self.load_router_exemplars()
        self._train_router()
        print("EmbeddingRouter initialized with weak model:", self.weak_model, "and strong model:", self.strong_model)

    def load_router_exemplars(self,filepath="phase2_routing/evaluation/train_prompts.json"):
        with open(filepath, "r") as file:
            data = json.load(file)

        exemplar_dict = {}
        
        for model_name, items in data["routes"].items():
            exemplar_dict[model_name] = [item["prompt"] for item in items]
            
        return exemplar_dict

    def _train_router(self):
        X = []
        y = []
        for model, prompts in self.routes.items():
            for prompt in prompts:
                embedding = self.embedding_model.encode(prompt)
                X.append(embedding)
                y.append(model)
        self.knn.fit(X, y)
    
    def route(self, query):
        start = time.time()
        query_embedding = self.embedding_model.encode(query).reshape(1, -1)
        predicted_model = self.knn.predict(query_embedding)[0]
        end = time.time()
        print(f"Routing query: '{query}' to model: '{predicted_model}' (Time: {end - start:.2f}s)")
        return {
            "model": predicted_model,
            "latency": round(end - start, 2)
        }

if __name__ == "__main__":
    router = EmbeddingRouter()
    test_query = "explain quantum entanglement"
    router.route(test_query)
