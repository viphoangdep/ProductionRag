"""Cổng vào vector store: giữ client để đóng khi app tắt."""


class VectorStore:
    """Vector store (Qdrant) + client. Đổi backend = sửa provider trong core/container.py."""

    def __init__(self, url: str, collection: str, embedding):
        from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url)
        self.store = QdrantVectorStore(
            client=self.client, collection_name=collection, embedding=embedding,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.HYBRID, vector_name="dense",
            sparse_vector_name="sparse",
            content_payload_key="page_content", metadata_payload_key="metadata")

    def close(self) -> None:
        self.client.close()
