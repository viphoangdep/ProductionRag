"""Infra adapters: Qdrant + CrossEncoder. Pure logic nằm ở tools/search_tool.py."""


def build_ancestor_filter(node_id: str):
    """Qdrant filter lay ca subtree (dung kem hybrid_search)."""
    from qdrant_client import models  # lazy: heavy dep only at runtime

    return models.Filter(
        should=[
            models.FieldCondition(key="metadata.section_id", match=models.MatchValue(value=node_id)),
            models.FieldCondition(key="metadata.ancestor_ids", match=models.MatchValue(value=node_id)),
        ]
    )


def hybrid_search(vector_store, query: str, k: int, qdrant_filter=None) -> list:
    from qdrant_client import models  # lazy: heavy dep only at runtime

    return vector_store.similarity_search_with_score(
        query=query, k=k, filter=qdrant_filter,
        hybrid_fusion=models.FusionQuery(fusion=models.Fusion.DBSF),
    )


class QdrantRetriever:
    """Retriever adapter trên hybrid Qdrant vector store. Đổi ở core/container.py."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query: str, k: int, qdrant_filter=None) -> list:
        return hybrid_search(self.vector_store, query, k, qdrant_filter)


class CrossEncoderReranker:
    """Reranker adapter trên CrossEncoder model. Đổi ở core/container.py."""

    def __init__(self, reranker):
        self.reranker = reranker

    def rerank(self, query_norm: str, docs: list) -> list:
        return rerank(self.reranker, query_norm, docs)


def rerank(reranker, query_norm: str, docs: list) -> list:
    pairs = [(query_norm, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
