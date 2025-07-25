# setup_qdrant.py
from qdrant_client import QdrantClient, models

# --- Constants ---
COLLECTION_NAME = "customer_support_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_SIZE = 384 # As per BAAI/bge-small-en-v1.5

def setup_qdrant_collection():
    """Initializes the Qdrant client and creates the collection if it doesn't exist."""
    client = QdrantClient(host="localhost", port=6333)

    # Check if collection already exists
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Creating it now.")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_SIZE,
                distance=models.Distance.COSINE
            ),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    
    return client

if __name__ == "__main__":
    qdrant_client = setup_qdrant_collection()
    print("Qdrant setup complete.")