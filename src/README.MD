ai-service/ — RAG chatbot API (FastAPI). Run from `src/` (package root).
│
├── src/                      # cd here to run: uvicorn main:app
│   ├── .env                  # source of truth for configs (model tiers, keys, pipeline params)
│   ├── main.py               # entrypoint: router + middleware wiring (web/deps.py)
│   │
│   ├── web/
│   │   ├── deps.py           # composition root: provider -> PORT
│   │   ├── controller/
│   │   │   ├── chat_controller.py   # POST /api/v1/chat/query, GET /assets/images/{id}
│   │   │   └── health_controller.py # GET /health
│   │   └── middleware/
│   │       ├── logging_middleware.py
│   │       └── exception_handler.py
│   │
│   ├── service/              # business logic, depends only on Ports
│   │   ├── ports.py          # 4 interfaces: Retriever / LLM / Reranker / AssetStore
│   │   ├── chat_service.py   # orchestrator: normalize -> hybrid+HyDE -> RRF -> parents -> rerank -> LLM
│   │   ├── rag_service.py    # pure fns (rrf_fuse, aggregate_parents, expand, context) + adapters
│   │   ├── llm_service.py    # HyDE paraphrases (cheap lv1) + answer (strong lv2), via init_chat_model
│   │   └── asset_service.py  # tables inline JSON + images proxy S3
│   │
│   ├── mapper/
│   │   └── chat_mapper.py
│   │
│   ├── enums/
│   │   └── search_type.py
│   │
│   ├── model/
│   │   ├── req/              # API Requests (chat_req.py)
│   │   ├── res/              # API Responses (chat_res.py + citations/tables/images)
│   │   └── domain/           # Internal domain objects
│   │
│   ├── config/
│   │   └── settings.py       # loads .env (LLM_LV1/LV2 + *_API_KEY, EMBED_*, HYDE_K, TOP_N, ...)
│   │
│   └── util/
│       ├── string_util.py    # normalize_ws, dedupe_paraphrases
│       └── json_util.py
│
├── tests/
│   ├── web/                  # test_chat_api.py
│   └── service/              # test_rag_logic / test_asset_service / test_model_tiers / test_settings_env
│
├── output/                   # chunks/tables/images metadata (local, AssetService loads at startup)
├── Dockerfile
├── docker-compose.yml        # Qdrant
├── pyproject.toml            # pythonpath=["src"] for running pytest from root
└── README.md

Run:
  docker compose up -d                    # Qdrant :6333 (run from root)
  ollama pull qwen3:1.7b                  # + model specified in src/.env
  cd src
  uvicorn main:app --host 0.0.0.0 --port 8000
  # test from root: python -m pytest tests -q

Switch models/providers (local <-> cloud): modify spec `provider:model` + key in src/.env.
