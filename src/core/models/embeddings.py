"""Factory embedding model (vector store dùng để embed)."""
from core.models.chat_model import timeout_kwarg


def create_embeddings(spec: str, api_key: str = "", timeout: int | None = None):
    from langchain.embeddings import init_embeddings  # lazy: nặng, chỉ load lúc warmup

    kw = {}
    if timeout:
        kw.update(timeout_kwarg(spec, timeout))
    try:
        if api_key:
            return init_embeddings(spec, api_key=api_key, **kw)
        return init_embeddings(spec, **kw)
    except (TypeError, ValueError):
        # provider local (VD: OllamaEmbeddings) không nhận api_key -> thử lại không key
        return init_embeddings(spec, **kw)
