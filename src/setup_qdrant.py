from qdrant_client import QdrantClient, models

DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_EMBEDDING_SIZE = 384
SPARSE_EMBEDDING_MODEL_NAME = "prithivida/Splade_PP_en_v1"
client = QdrantClient(host="localhost", port=6333)

def create_or_recreate_collection(name: str, indexes: dict, use_hnsw_optimization: bool = False):
    """Creates a new Qdrant collection with advanced options."""
    try:
        client.get_collection(collection_name=name)
        print(f"Collection '{name}' already exists. Recreating it for a clean slate.")
        client.delete_collection(collection_name=name)
    except Exception:
        pass  # Collection does not exist, which is fine

    print(f"Creating collection '{name}'...")
    
    hnsw_config = None
    if use_hnsw_optimization:
        print(f"Applying HNSW optimization for collection '{name}'.")
        hnsw_config = models.HnswConfigDiff(
            payload_m=16,
            m=0,
            on_disk=True,
        )

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_EMBEDDING_SIZE,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
        hnsw_config=hnsw_config,
        on_disk_payload=True,
    )
    
    print(f"Creating payload indexes for '{name}'...")
    for field, field_type in indexes.items():
        # The field_schema now includes the is_tenant flag
        client.create_payload_index(name, field, field_schema=field_type)
    print(f"Collection '{name}' created successfully.")


if __name__ == "__main__":
    user_data_indexes = {
        "tenant_id": models.KeywordIndexParams(
            type='keyword',
            is_tenant=True,
            on_disk=True
        ),
        "customer_id": models.KeywordIndexParams(
            type='keyword',
            is_tenant=True,
            on_disk=True
        )
    }

    kb_indexes = {
        "tenant_id": models.KeywordIndexParams(
            type='keyword',
            is_tenant=True,
            on_disk=True
        ),
        "tags": models.KeywordIndexParams(
            type='keyword',
            on_disk=True
        ),
        "source_type": models.KeywordIndexParams(
            type='keyword',
            on_disk=True
        )
    }

    # Create collections with the new advanced settings
    create_or_recreate_collection("user_data", indexes=user_data_indexes, use_hnsw_optimization=True)
    create_or_recreate_collection("knowledge_base", indexes=kb_indexes, use_hnsw_optimization=True)
    print("Qdrant setup complete.")