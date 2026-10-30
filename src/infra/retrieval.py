"""Infra adapters: Qdrant + CrossEncoder. Translates provider-agnostic filters to Qdrant."""
from __future__ import annotations

from typing import Any


def build_ancestor_filter(node_id: str):
    """Qdrant filter lay ca subtree (dung kem hybrid_search)."""
    from qdrant_client import models  # lazy: heavy dep only at runtime

    return models.Filter(
        should=[
            models.FieldCondition(key="metadata.section_id", match=models.MatchValue(value=node_id)),
            models.FieldCondition(key="metadata.ancestor_ids", match=models.MatchValue(value=node_id)),
        ]
    )


def to_qdrant_filter(filter_spec: Any):
    """Translates provider-agnostic SearchFilter, dict, or models.Filter into Qdrant models.Filter."""
    if filter_spec is None:
        return None

    from qdrant_client import models

    if isinstance(filter_spec, models.Filter):
        return filter_spec

    must_conditions = []
    should_conditions = []

    node_id = getattr(filter_spec, "node_id", None)
    if node_id:
        should_conditions.extend([
            models.FieldCondition(key="metadata.section_id", match=models.MatchValue(value=node_id)),
            models.FieldCondition(key="metadata.ancestor_ids", match=models.MatchValue(value=node_id)),
        ])

    meta = getattr(filter_spec, "metadata", None)
    if meta is None and isinstance(filter_spec, dict):
        meta = filter_spec
    elif meta is None:
        meta = {}

    for k, v in meta.items():
        must_conditions.append(models.FieldCondition(key=f"metadata.{k}", match=models.MatchValue(value=v)))

    if must_conditions and should_conditions:
        return models.Filter(must=must_conditions, should=should_conditions)
    if must_conditions:
        return models.Filter(must=must_conditions)
    if should_conditions:
        return models.Filter(should=should_conditions)
    return None


def hybrid_search(vector_store, query: str, k: int, qdrant_filter=None) -> list:
    from qdrant_client import models  # lazy: heavy dep only at runtime

    return vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=qdrant_filter,
        hybrid_fusion=models.FusionQuery(fusion=models.Fusion.DBSF),
    )


class QdrantRetriever:
    """Retriever adapter trên hybrid Qdrant vector store. Đổi ở core/container.py."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query: str, k: int, filter=None, qdrant_filter=None) -> list:
        active_filter = filter if filter is not None else qdrant_filter
        q_filter = to_qdrant_filter(active_filter)
        return hybrid_search(self.vector_store, query, k, q_filter)


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
