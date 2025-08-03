import json
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    Prefetch, SparseVector, FusionQuery, Fusion,
)


dense_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

def retrieve_context(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    tenant_id: str,
    source_type: str = None,
    tags: list[str] = None,
    customer_id: str = None,
    k_prefetch: int = 10,
    top_k: int = 5,
    fusion_method: Fusion = Fusion.RRF,
):
    """
    Retrieve the top-K most semantically similar points matching the given filters.
    """

    # 1. Embed the query string
    dense_vector = list(dense_embedding_model.embed([query_text]))[0]
    sparse_result = list(sparse_embedding_model.embed([query_text]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices,
        values=sparse_result.values,
    )

    # 2. Build payload filter conditions
    must_clauses = [
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
    ]
    if source_type:
        must_clauses.append(
            FieldCondition(key="source_type", match=MatchValue(value=source_type))
        )
    if customer_id:
        must_clauses.append(
            FieldCondition(key="customer_id", match=MatchValue(value=customer_id))
        )
    if tags:
        # Match any of the supplied tags
        must_clauses.append(
            FieldCondition(
                key="tags",
                match=MatchAny(any=tags)  # Qdrant supports list match for keyword
            )
        )

    payload_filter = Filter(must=must_clauses)

    # 3. Define prefetch queries
    prefetches = [
        Prefetch(
            query=sparse_vec,
            using="sparse",
            limit=k_prefetch,
        ),
        Prefetch(
            query=dense_vector.tolist() if isinstance(dense_vector, (list, tuple)) else dense_vector,
            using="dense",
            limit=k_prefetch,
        ),
    ]

    # 4. Build the fusion query
    fusion_query = FusionQuery(fusion=fusion_method)

    results = client.query_points(
        collection_name=collection_name,
        prefetch=prefetches,
        query=fusion_query,
        query_filter=payload_filter,
        limit=top_k,
        with_payload=True
       
    )

    # 4. Return id, score and payload for each hit
    return [
        {
            "id": hit.id,
            'similarity_with_query': hit.score,
            "payload": hit.payload
        }
        for hit in results.points
    ]

def retrieve_customer_info(
    client: QdrantClient,
    tenant_id: str,
    customer_id: str,
):
    must_clauses = [
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
    ]
    must_clauses.append(
        FieldCondition(key="source_type", match=MatchValue(value="crm"))
    )
    must_clauses.append(
        FieldCondition(key="customer_id", match=MatchValue(value=customer_id))
    )

    payload_filter = Filter(must=must_clauses)

    results, _ = client.scroll(
        collection_name="user_data",
        scroll_filter=payload_filter,
        limit=1,
    )

    if not results:
        return f"No customer information found for this tenant_id: {tenant_id} and customer_id: {customer_id}."
    
    return results[0].payload

def retrieve_customer_helpdesk_logs(client: QdrantClient, query: str, customer_id: str, tenant_id: str, top_k: int = 3, k_prefetch: int = 10) -> str:
    """
    Retrieves a comprehensive context for a user by fetching data from
    user_data (CRM, helpdesk) and knowledge_base collections.
    """

    helpdesk_records = retrieve_context(
        client=client,
        collection_name="user_data",
        query_text=query,
        tenant_id=tenant_id,
        source_type="helpdesk",
        customer_id=customer_id,
        top_k=top_k,
        k_prefetch = k_prefetch,
        fusion_method = Fusion.RRF,
    )

    if not helpdesk_records:
        return f"No relevant customer helpdesk ticket found for this tenant_id: {tenant_id} and customer_id: {customer_id} for this particular query."

    if helpdesk_records:
        sanitized_records = [{k: v for k, v in record.items() if not k == 'id'} for record in helpdesk_records]
        context = json.dumps(sanitized_records, indent=2)
        
    return context

def retrieve_related_knowledge_base(client: QdrantClient, query: str, tenant_id: str, source_type: str, tags: list=None, top_k: int = 3, k_prefetch: int = 10) -> str:
    related_kb = retrieve_context(
        client=client,
        collection_name="knowledge_base",
        query_text=query,
        tenant_id=tenant_id,
        source_type=source_type,
        tags=tags,
        top_k=top_k,
        k_prefetch = k_prefetch,
        fusion_method = Fusion.RRF,
    )

    if not related_kb:
        return f"No relevant knowledge base found for tenant_id: {tenant_id}, source_type: {source_type} with tags: {tags} for this particular query"
    
    sanitized_records = [{k: v for k, v in doc.items() if not k == 'id'} for doc in related_kb]
    context = json.dumps(sanitized_records, indent=2)
    return context
