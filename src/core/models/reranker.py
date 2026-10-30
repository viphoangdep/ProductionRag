"""Factory reranker model (CrossEncoder)."""


def create_reranker_model(spec: str):
    from sentence_transformers import CrossEncoder  # lazy: kéo TF, chỉ load lúc warmup

    return CrossEncoder(spec)
