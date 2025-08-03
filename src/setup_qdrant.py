from qdrant_client import QdrantClient, models

DENSE_EMBEDDING_SIZE = 384
IMAGE_EMBEDDING_SIZE = 512
client = QdrantClient(host="localhost", port=6333)

def create_or_recreate_collection(
    name: str, 
    indexes: dict, 
    use_sparse: bool = True, 
    use_image: bool = False,
    use_hnsw_optimization: bool = False,
    distance_metric: models.Distance = models.Distance.COSINE
):
    """Creates a new Qdrant collection with advanced options."""
    try:
        client.get_collection(collection_name=name)
        print(f"Collection '{name}' already exists. Recreating it for a clean slate.")
        client.delete_collection(collection_name=name)
    except Exception:
        pass

    print(f"Creating collection '{name}'...")
    
    hnsw_config = None
    if use_hnsw_optimization:
        print(f"Applying HNSW optimization for collection '{name}'.")
        hnsw_config = models.HnswConfigDiff(
            payload_m=16,
            m=0,
            on_disk=True,
        )

    sparse_vectors_config = None
    if use_sparse:
        sparse_vectors_config = {
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True
                )
            )
        }

    vectors_config={
        "dense": models.VectorParams(
            size=DENSE_EMBEDDING_SIZE,
            distance=distance_metric,
            on_disk=True,
        ),
    }
    if use_image:
        vectors_config['image'] = models.VectorParams(
            size=IMAGE_EMBEDDING_SIZE,
            distance=distance_metric,
            on_disk=True,   
        )

    client.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
        hnsw_config=hnsw_config,
        on_disk_payload=True,
    )
    
    print(f"Creating payload indexes for '{name}'...")
    for field, field_type in indexes.items():
        client.create_payload_index(name, field, field_schema=field_type)
    print(f"Collection '{name}' created successfully.")


if __name__ == "__main__":
    user_data_indexes = {
        "tenant_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True),
        "customer_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True)
    }

    kb_indexes = {
        "tenant_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True),
        "tags": models.KeywordIndexParams(type='keyword', on_disk=True),
        "source_type": models.KeywordIndexParams(type='keyword', on_disk=True)
    }

    create_or_recreate_collection("user_data", indexes=user_data_indexes, use_hnsw_optimization=True)
    create_or_recreate_collection("knowledge_base", indexes=kb_indexes, use_hnsw_optimization=True)


    cache_indexes = {
        "tenant_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True),
        "customer_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True)
    }
    create_or_recreate_collection("semantic_cache", indexes=cache_indexes, use_sparse=False, distance_metric=models.Distance.EUCLID)

    order_indexes = {
        "tenant_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True),
        "customer_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True),
        "order_id": models.KeywordIndexParams(type='keyword', is_tenant=True, on_disk=True)
    }
    create_or_recreate_collection("orders", indexes=order_indexes, use_image=True)
    print("Qdrant setup complete.")