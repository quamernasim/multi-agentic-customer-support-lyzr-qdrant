import os
import uuid
import json
import pandas as pd
from qdrant_client import QdrantClient
from tqdm import tqdm
from qdrant_client.models import PointStruct, SparseVector
from fastembed import SparseTextEmbedding, TextEmbedding

DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_EMBEDDING_MODEL_NAME = "prithivida/Splade_PP_en_v1"
DENSE_EMBEDDING_MODEL = TextEmbedding(model_name=DENSE_EMBEDDING_MODEL_NAME)
SPARSE_EMBEDDING_MODEL = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL_NAME)
client = QdrantClient(host="localhost", port=6333)



def process_unstructured_files(directory_path, tenant_id, points_list):
    """Dynamically reads all JSON files from a given directory."""
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            source_name = filename.split('.')[0]  # e.g., 'faqs', 'policy'
            
            with open(filepath, 'r') as f:
                data = json.load(f)

                for item in data:
                    category = item.get(
                        "question", item.get("title", {})
                    )
                    content = item.get(
                        "answer", item.get(
                            "content", item.get("description", "")
                        )
                    )
                    tags = item.get("tags", {})
                    
                    if not content:
                        continue

                    text_to_embed = content
                    # THIS CAN ONLY BE QUESTION/TITLE/POLICY_TYPE
                    if source_name == 'faqs':
                        text_to_embed = f"Question: {category}\nAnswer: {content}"
                    elif source_name == 'handbook':
                        text_to_embed = f"Title: {category}\nContent: {content}"
                    elif source_name == 'policy':
                        text_to_embed = f"Policy Type: {category}\nPolicy Description: {content}"

                    payload = {
                        "tenant_id": tenant_id,
                        "source_type": source_name,
                        "tags": tags,
                        "content": text_to_embed,
                    }
                    points_list.append((text_to_embed, payload))


def upsert_in_batch(text, payloads, collection_name, batch_size):
    dense = list(DENSE_EMBEDDING_MODEL.embed(text))
    sparse = list(SPARSE_EMBEDDING_MODEL.embed(text))

    total = len(text)
    for i in tqdm(range(0, total, batch_size)):
        batch_dense_embeddings = dense[i:i + batch_size]
        batch_sparse_embeddings = sparse[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embeddings,
                    "sparse": SparseVector(indices=sparse_embeddings.indices, values=sparse_embeddings.values)
                },
                payload=payload
            )
            for dense_embeddings, sparse_embeddings, payload in zip(batch_dense_embeddings, batch_sparse_embeddings, batch_payloads)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )

def ingest_data(data_path, batch_size = 64):
    """Processes all data sources and uploads them to their respective collections."""
    user_data_points = []
    kb_points = []

    tenants = ["ecom", "fintech"]
    for tenant in tenants:
        print(f"\n--- Processing data for tenant: {tenant} ---")
        
        crm_df = pd.read_csv(f"{data_path}/{tenant}/crm_records.csv")
        for _, row in crm_df.iterrows():
            text_to_embed = f"Customer: {row['name']}, Email: {row['email']}"
            payload = {"tenant_id": tenant, "source_type": "crm", **row.to_dict(), "text_embeded": text_to_embed}
            user_data_points.append((text_to_embed, payload))

        helpdesk_df = pd.read_csv(f"{data_path}/{tenant}/helpdesk_logs.csv")
        for _, row in helpdesk_df.iterrows():
            text_to_embed = f"Ticket: {row['issue_summary']}, Status: {row['status']}"
            payload = {"tenant_id": tenant, "source_type": "helpdesk", **row.to_dict(), "text_embeded": text_to_embed}
            user_data_points.append((text_to_embed, payload))

        kb_path = f"{data_path}/{tenant}/knowledge_base"
        process_unstructured_files(kb_path, tenant, kb_points)

    # --- Upload to Qdrant ---
    # User Data
    user_data_texts, user_data_payloads = zip(*user_data_points)
    # user_data_embeddings = list(DENSE_EMBEDDING_MODEL.embed(user_data_texts))[0]
    upsert_in_batch(user_data_texts, user_data_payloads, "user_data", batch_size)
    print(f"\nIngested {len(user_data_points)} points into 'user_data' collection.")

    # Knowledge Base
    kb_texts, kb_payloads = zip(*kb_points)
    # kb_embeddings = list(embedding_model.embed(kb_texts))
    upsert_in_batch(kb_texts, kb_payloads, "knowledge_base", batch_size)
    print(f"Ingested {len(kb_points)} points into 'knowledge_base' collection.")


if __name__ == "__main__":
    ingest_data(data_path = '../data')
