"""Composition root: khai báo toàn bộ service/instance ở đây.

Provider tự gọi constructor -> không cần hàm _build_* riêng.
"""
from __future__ import annotations

import logging
import time

from dependency_injector import containers, providers

from config.settings import settings
from core.agent import create_agent, create_agent_context
from core.database.assets import AssetService
from core.database.vector_store import VectorStore
from core.grounding.verifier import GroundingVerifier
from core.models.chat_model import create_chat_model
from core.models.embeddings import create_embeddings
from core.models.reranker import create_reranker_model
from core.router import QueryRouter
from core.session.analyzer import ConversationAnalyzer
from core.session.session_manager import SessionManager
from infra.retrieval import CrossEncoderReranker, QdrantRetriever
from infra.section_store import SectionStore
from service.chat_service import ChatService
from service.corpus_service import CorpusService

log = logging.getLogger("ai-service")


def create_session_store():
    """Factory creating SessionStore based on configured backend (memory vs postgres)."""
    if getattr(settings, "session_store_backend", "memory").lower() == "postgres":
        from infra.postgres_session_store import PostgresSessionStore

        return PostgresSessionStore(
            connection_string=settings.postgres_url,
            table_name=settings.postgres_table_name,
            max_history_turns=settings.max_history_turns,
        )
    return SessionManager(
        ttl_minutes=settings.session_ttl_minutes,
        max_sessions=settings.max_sessions,
        max_history_turns=settings.max_history_turns,
    )


class Container(containers.DeclarativeContainer):
    """Mỗi provider = 1 bean. Singleton = dùng chung, build đúng 1 lần."""

    # --- model ---
    weak_model = providers.Singleton(create_chat_model, spec=settings.llm_lv1,
                                    api_key=settings.llm_lv1_api_key,
                                    timeout=settings.llm_timeout)
    strong_model = providers.Singleton(create_chat_model, spec=settings.llm_lv2,
                                      api_key=settings.llm_lv2_api_key,
                                      timeout=settings.llm_timeout)
    reranker_model = providers.Singleton(create_reranker_model, spec=settings.reranker_model)
    embeddings = providers.Singleton(create_embeddings, spec=settings.embed_model,
                                     api_key=settings.embed_api_key,
                                     timeout=settings.llm_timeout)

    # --- database ---
    vector_store = providers.Singleton(VectorStore, url=settings.qdrant_url,
                                       collection=settings.collection, embedding=embeddings)

    # --- adapter ---
    retriever = providers.Singleton(QdrantRetriever, vector_store=vector_store.provided.store)
    reranker = providers.Singleton(CrossEncoderReranker, reranker=reranker_model)
    sections = providers.Singleton(SectionStore.from_output_dir, str(settings.output_path))
    asset_store = providers.Singleton(AssetService.from_output_dir, str(settings.output_path))
    session_manager = providers.Singleton(create_session_store)
    grounding_verifier = providers.Singleton(GroundingVerifier,
                                            weak_model=weak_model,
                                            enable_eval=settings.enable_grounding_eval,
                                            threshold=settings.grounding_score_threshold)
    conversation_analyzer = providers.Singleton(ConversationAnalyzer, weak_model=weak_model)
    query_router = providers.Singleton(QueryRouter, weak_model=weak_model)

    # --- agent ---
    agent = providers.Singleton(create_agent, weak_model,
                              checkpointer=session_manager.provided.checkpointer)
    agent_context = providers.Singleton(create_agent_context, retriever=retriever,
                                        reranker=reranker, assets=asset_store,
                                        sections=sections,
                                        weak_model=weak_model, strong_model=strong_model,
                                        router=query_router)

    # --- service (tầng nghiệp vụ) ---
    chat_service = providers.Singleton(ChatService, agent=agent, agent_context=agent_context,
                                       weak_model=weak_model,
                                       session_manager=session_manager,
                                       grounding_verifier=grounding_verifier,
                                       conversation_analyzer=conversation_analyzer,
                                       n_paraphrases=settings.n_paraphrases)
    corpus_service = providers.Singleton(CorpusService, cfg=settings)


container = Container()

_warmed = False  # True sau warmup: biết client đã build để close


def close() -> None:
    """Đóng client vector store khi app tắt. Không build mới nếu chưa warmup."""
    if _warmed:
        container.vector_store().close()


def warmup() -> dict:
    """Build toàn bộ bean theo thứ tự rẻ→đắt. Fail-fast: lỗi là raise để crash."""
    global _warmed
    # Dòng này để check env có ăn không: chỉ log TÊN + có key hay không, không log key.
    log.info("config weak=%s strong=%s embed=%s qdrant=%s collection=%s "
             "keys(lv1/lv2/embed)=%s/%s/%s n_paraphrases=%d",
             settings.llm_lv1, settings.llm_lv2, settings.embed_model,
             settings.qdrant_url, settings.collection,
             bool(settings.llm_lv1_api_key), bool(settings.llm_lv2_api_key),
             bool(settings.embed_api_key), settings.n_paraphrases)
    for name, prov in container.providers.items():
        t = time.perf_counter()
        bean = prov()
        extra = ""
        if name in ("weak_model", "strong_model"):
            extra = f" type={type(bean).__name__} model={getattr(bean, 'model_name', getattr(bean, 'model', '?'))}"
        log.info("warmup %s ok %.0fms%s", name, (time.perf_counter() - t) * 1000, extra)
    _warmed = True
    return {"state": "ready", "detail": f"models={settings.llm_lv1}/{settings.llm_lv2}"}
