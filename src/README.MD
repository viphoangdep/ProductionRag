# AI-Service — Production RAG Chatbot API (FastAPI)

Production-grade RAG backend utilizing FastAPI, LangChain, LangGraph, Qdrant Hybrid Search (Dense + BM25 Sparse), RRF Fusion, Hierarchical Section Expansion, CrossEncoder reranking, and Grounding evaluation.

## Project Structure

```
├── pyproject.toml              # Dependencies & pytest configuration
├── Dockerfile
├── docker-compose.yml          # Qdrant service
├── output/                     # Corpus metadata (chunks, tables, images, section_tree)
├── tests/                      # Automated test suite
│   ├── test_asset_service.py
│   ├── test_chat_api.py
│   ├── test_chat_service.py
│   ├── test_grounding.py
│   ├── test_multiturn_chat.py
│   ├── test_output_format.py
│   └── test_security_and_budget.py
└── src/
    ├── main.py                 # FastAPI application entrypoint & middleware setup
    ├── .env.example            # Configuration template
    ├── config/
    │   ├── settings.py         # Pydantic BaseSettings (models, tiers, paths, thresholds)
    │   └── prompts.py          # System, HyDE, Grounding, and Context Analyzer prompts
    ├── core/
    │   ├── agent.py            # LangGraph agent factory & context builder
    │   ├── container.py        # Dependency injection container & bean warmup
    │   ├── router.py           # Dynamic routing (simple queries vs RAG)
    │   ├── security.py         # Prompt injection detector & tool call budget tracker
    │   ├── database/           # Vector store & corpus asset resolvers
    │   │   ├── vector_store.py # QdrantVectorStore connection & management
    │   │   └── assets.py       # Corpus tables & images resolver from output/
    │   ├── exception/
    │   │   └── errors.py       # Domain AppError & ErrorCode
    │   ├── grounding/
    │   │   └── verifier.py     # GroundingVerifier (anti-hallucination & citations)
    │   ├── models/             # Model factory providers
    │   │   ├── chat_model.py   # Chat LLM factory (Ollama, OpenAI, Gemini, etc.)
    │   │   ├── embeddings.py   # Embedding model factory
    │   │   └── reranker.py     # CrossEncoder reranker factory
    │   └── session/
    │       ├── analyzer.py     # ConversationAnalyzer (topic relation & rewrite)
    │       └── session_manager.py # Multi-turn session manager with TTL & LRU eviction
    ├── infra/
    │   ├── retrieval.py        # QdrantRetriever adapter & CrossEncoder reranker
    │   └── section_store.py    # Hierarchical section tree resolver
    ├── model/
    │   ├── req/chat_req.py     # ChatQueryRequest DTO
    │   └── res/chat_res.py     # ChatQueryResponse, Citation, TableRef, ImageRef DTOs
    ├── service/
    │   └── chat_service.py     # Application use-case orchestrator
    ├── tools/
    │   ├── search_tool.py      # LangChain tool for document retrieval
    │   └── errors.py           # Tool error handling middleware
    ├── util/
    │   └── string_util.py      # Text normalization & paraphrase deduplication
    └── web/
        ├── deps.py             # FastAPI Depends providers
        ├── controller/
        │   ├── chat_controller.py   # /api/v1/chat endpoints
        │   └── health_controller.py # /health & /ready endpoints
        └── middleware/
            ├── logging_middleware.py
            ├── rate_limit_middleware.py
            ├── timeout_middleware.py
            └── exception_handler.py
```

## Quickstart

```powershell
# 1. Start Qdrant vector database
docker compose up -d

# 2. Run API service
cd src
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Run test suite
python -m pytest tests -vv
```
