from pathlib import Path

from pydantic_settings import BaseSettings

PKG_ROOT = Path(__file__).resolve().parents[1]   # src/
REPO_ROOT = PKG_ROOT.parent                        # repo root


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    collection: str = "rag_docs"
    embed_model: str = "ollama:RizwanMalik/qwen3-vl-embedding-2b:q4_0-q8_0"
    embed_api_key: str = ""
    llm_lv1: str = "ollama:qwen3:1.7b"  # cheap tier: HyDE paraphrases
    llm_lv1_api_key: str = ""
    llm_lv2: str = "ollama:qwen3:1.7b"  # strong tier: final answer (higher lv = stronger)
    llm_lv2_api_key: str = ""
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    llm_timeout: int = 60  # giây; treo lâu hơn -> trả 5xx thay vì giữ thread
    n_paraphrases: int = 3
    direct_k: int = 10
    evidence_top1_thr: float = 0.5
    evidence_mean_thr: float = 0.3
    hyde_k: int = 10
    fused_k: int = 20
    rrf_k: int = 60
    top_parents: int = 3
    max_child_per_parent: int = 10
    top_n: int = 5
    output_dir: str = "output"
    max_table_rows: int = 50
    session_ttl_minutes: int = 60
    max_sessions: int = 1000
    max_history_turns: int = 6
    enable_grounding_eval: bool = True
    grounding_score_threshold: float = 0.7
    # Rate limit & Timeout & Security guards
    rate_limit_requests: int = 30  # Max requests per window
    rate_limit_window_seconds: int = 60  # Sliding window duration
    request_timeout_seconds: int = 60  # Max agent/query execution duration
    enable_injection_guard: bool = True  # Block prompt injections
    tool_call_budget: int = 3  # Max search_documents calls per query

    @property
    def output_path(self) -> Path:
        """Corpus luôn nằm ở repo root, không phụ thuộc thư mục đang chạy."""
        p = Path(self.output_dir)
        return p if p.is_absolute() else (REPO_ROOT / p)

    # env_file tuyệt đối: luôn đọc src/.env, chạy từ đâu cũng giống nhau.
    model_config = {"env_file": PKG_ROOT / ".env", "extra": "ignore"}


settings = Settings()
