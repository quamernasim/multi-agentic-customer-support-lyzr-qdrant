# multi-agentic-customer-support-lyzr-qdrant
An enterprise-grade, multi-agent customer support system built with the Lyzr agentic framework and Qdrant vector database. Demonstrates advanced RAG, hybrid search, and multimodal capabilities using open-source models.



docker run --gpus=all -p 6333:6333 -p 6334:6334 -e QDRANT__GPU__INDEXING=1 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:gpu-nvidia-latest

