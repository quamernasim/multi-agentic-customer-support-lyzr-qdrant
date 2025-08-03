# src/cache.py
import uuid
import time
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from qdrant_client.models import PointStruct, NamedVector

DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_EMBEDDING_MODEL = TextEmbedding(model_name=DENSE_EMBEDDING_MODEL_NAME)
client = QdrantClient(host="localhost", port=6333)

class SemanticCache:
    def __init__(self, threshold: float = 0.2):
        self.client = client
        self.embedding_model = DENSE_EMBEDDING_MODEL
        self.collection_name = "semantic_cache"
        self.threshold = threshold 

    def check_cache(self, query_text: str, tenant_id: str, customer_id: str):
        """Checks the cache for a semantically similar query."""
        start_time = time.time()
        query_vector = list(self.embedding_model.embed([query_text]))[0]

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedVector(
                name="dense",
                vector=query_vector
            ),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                    models.FieldCondition(key="customer_id", match=models.MatchValue(value=customer_id))
                ]
            ),
            limit=1,
            with_payload=True
        )
        if search_result and search_result[0].score <= self.threshold:
            end_time = time.time()
            print(f"CACHE HIT! (Score: {search_result[0].score:.4f}, Time: {end_time - start_time:.4f}s)")
            return search_result[0].payload.get("response")
        
        print("CACHE MISS!")
        return None

    def add_to_cache(self, query_text: str, response_text: str, tenant_id: str, customer_id: str):
        """Adds a new query-response pair to the cache."""
        query_vector = list(self.embedding_model.embed([query_text]))[0]

        points=PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": query_vector,
            },
            payload={
                'response': response_text,
                'tenant_id': tenant_id,
                'customer_id': customer_id,
            },
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[points],
            wait=True
        )
        print("Added new entry to semantic cache.")