"""Tests for core interfaces (BaseRetriever, BaseReranker, etc.) and provider interchangeability."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from core.interfaces.asset_store import BaseAssetStore
from core.interfaces.expander import BaseQueryExpander
from core.interfaces.reranker import BaseReranker
from core.interfaces.retriever import BaseRetriever
from core.interfaces.section_store import BaseSectionStore
from core.interfaces.session_store import BaseSessionStore
from core.interfaces.verifier import BaseGroundingVerifier
from core.rag.pipeline import RAGPipeline
from core.rag.types import RAGRuntimeContext, SearchFilter
from infra.retrieval import CrossEncoderReranker, QdrantRetriever, WeaviateRetriever


class DummyStore:
    def __init__(self, docs=None):
        self.docs = docs or [
            Document(page_content="Content 1", metadata={"chunk_id": "c1", "parent_id": "p1"}),
            Document(page_content="Content 2", metadata={"chunk_id": "c2", "parent_id": "p1"}),
        ]

    def similarity_search_with_score(self, query, k, **kwargs):
        return [(d, 0.85) for d in self.docs[:k]]


class DummyAssets:
    def resolve_for_chunk(self, metadata, max_rows=50):
        return {"images": [], "tables": []}

    def image_bytes(self, image_id):
        return (b"fake_image", "image/png")


class DummySections:
    nodes = {}

    def resolve(self, name):
        return "sec_1"

    def depth_of(self, doc):
        return 1

    def subtree_filter(self, node_id):
        return SearchFilter(node_id=node_id)


class DummyReranker(BaseReranker):
    def rerank(self, query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        return [(d, 0.9 - i * 0.1) for i, d in enumerate(docs)]


def test_interfaces_are_abstract():
    with pytest.raises(TypeError):
        BaseRetriever()  # type: ignore

    with pytest.raises(TypeError):
        BaseReranker()  # type: ignore

    with pytest.raises(TypeError):
        BaseSectionStore()  # type: ignore

    with pytest.raises(TypeError):
        BaseAssetStore()  # type: ignore

    with pytest.raises(TypeError):
        BaseQueryExpander()  # type: ignore

    with pytest.raises(TypeError):
        BaseGroundingVerifier()  # type: ignore

    with pytest.raises(TypeError):
        BaseSessionStore()  # type: ignore


def test_qdrant_retriever_subclasses_base_retriever():
    retriever = QdrantRetriever(DummyStore())
    assert isinstance(retriever, BaseRetriever)


def test_weaviate_retriever_subclasses_base_retriever():
    retriever = WeaviateRetriever(DummyStore())
    assert isinstance(retriever, BaseRetriever)


def test_weaviate_filter_translation():
    retriever = WeaviateRetriever(DummyStore())

    # Empty filter
    assert retriever._translate_filter(None) is None

    # Filter with node_id and metadata
    f = SearchFilter(node_id="section_42", metadata={"source_file": "manual.pdf"})
    w_filter = retriever._translate_filter(f)
    assert w_filter is not None
    assert w_filter["operator"] == "And"
    assert len(w_filter["operands"]) == 2

    # Check node condition (Or across section_id and ancestor_ids)
    or_cond = w_filter["operands"][0]
    assert or_cond["operator"] == "Or"
    assert or_cond["operands"][0]["valueText"] == "section_42"


def test_weaviate_retriever_search_executes():
    retriever = WeaviateRetriever(DummyStore())
    f = SearchFilter(node_id="sec_1")
    results = retriever.search("query", k=2, filter=f)
    assert len(results) == 2
    assert results[0][0].page_content == "Content 1"


def test_pipeline_swaps_retriever_seamlessly():
    """Verify RAGPipeline runs with WeaviateRetriever and custom BaseReranker without modifying any pipeline code."""
    weaviate_retriever = WeaviateRetriever(DummyStore())
    custom_reranker = DummyReranker()
    assets = DummyAssets()
    sections = DummySections()

    ctx = RAGRuntimeContext(
        retriever=weaviate_retriever,
        reranker=custom_reranker,
        assets=assets,
        sections=sections,
        cfg={"evidence_top1_thr": 0.5, "evidence_mean_thr": 0.3},
    )

    context_str, artifact = RAGPipeline.execute(
        query="test query for weaviate",
        section="chapter 1",
        metadata={"category": "spec"},
        ctx=ctx,
    )

    assert "Content 1" in context_str
    assert artifact["branch"] == "direct"
    assert len(artifact["citations"]) > 0
    assert artifact["citations"][0]["chunk_id"] == "c1"

