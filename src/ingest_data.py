# ingest_data.py
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from tqdm import tqdm
import uuid

# --- Constants ---
COLLECTION_NAME = "customer_support_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATASET_PATH = "data/Bitext_Sample_Customer_Service_Training_Dataset/Training/Bitext_Sample_Customer_Service_Training_Dataset.csv"

from qdrant_client.models import PointStruct
from tqdm import tqdm

def ingest_data_to_qdrant(batch_size=256):
    client = QdrantClient(host="localhost", port=6333)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Load and prepare data
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna()
    df['text_to_embed'] = df['utterance'] + " - intent: " + df['intent']
    documents = df['text_to_embed'].tolist()
    payloads = df[['utterance', 'category', 'intent']].to_dict('records')

    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = list(embedding_model.embed(documents))

    print(f"Upserting data into Qdrant collection in batches of {batch_size}...")
    total = len(embeddings)

    for i in tqdm(range(0, total, batch_size)):
        batch_embeddings = embeddings[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            for embedding, payload in zip(batch_embeddings, batch_payloads)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

    print("Data ingestion complete.")

if __name__ == "__main__":
    ingest_data_to_qdrant()
