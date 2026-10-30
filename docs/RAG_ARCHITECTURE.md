# Architecture & Workflow: RAG System

Tài liệu này mô tả chi tiết toàn bộ kiến trúc và luồng xử lý truy vấn (**Query Workflow**) của hệ thống RAG chatbot hiện tại, từ lúc nhận API request đến khi trả lời người dùng và kiểm tra tính trung thực (**Grounding Verification**).

---

## 1. Sơ đồ luồng tổng quan (End-to-End Pipeline)

```mermaid
flowchart TD
    Client["User / Client"] -->|POST /api/v1/chat/query| Controller["FastAPI Controller"]
    Controller --> Service["ChatService"]

    subgraph Step1 ["1. Session & Query Analysis"]
        Service --> SM["SessionManager (Get last 4 turns)"]
        SM --> Analyzer["ConversationAnalyzer"]
        Analyzer --> NeedsContext{"Needs Context?"}
        NeedsContext -->|Yes: CONTINUE| Rewrite["Rewrite to Standalone Query"]
        NeedsContext -->|No: NEW_TOPIC| DirectQuery["Use Original Query"]
    end

    subgraph Step2 ["2. Dynamic Router"]
        Rewrite --> Router["dynamic_router"]
        DirectQuery --> Router
        Router --> CheckSimple{"is_simple()?"}
        CheckSimple -->|Yes: Simple/Persona| WeakBranch["Weak LLM: gpt-4o-mini<br/>No Tools, SIMPLE_PROMPT"]
        CheckSimple -->|No: Domain/RAG| StrongBranch["Strong LLM: gpt-4o<br/>Tool: search_documents, SYSTEM_PROMPT"]
    end

    subgraph Step3 ["3. Retrieval Pipeline (search_documents)"]
        StrongBranch -.->|Call search tool| SearchTool["search_documents"]
        SearchTool --> DirectRetrieval["Stage 1: Direct Hybrid Search (k=10)"]
        DirectRetrieval --> Group1["Aggregate by Section / Parent"]
        Group1 --> Expand1["Top Parents Expansion"]
        Expand1 --> Rerank1["CrossEncoder Rerank (top_n=5)"]
        Rerank1 --> Gate{"Confidence Gate?<br/>top1 >= 0.5 or mean >= 0.3"}

        Gate -->|Pass| PackDirect["Pack Context & Citations (Direct)"]
        Gate -->|Fail| HyDEBranch["Stage 2: HyDE Paraphrasing (Weak LLM)"]
        HyDEBranch --> MultiRetrieve["Multi-query Hybrid Search (Base + Paraphrases)"]
        MultiRetrieve --> RRFFusion["RRF Fusion (Rank Merging)"]
        RRFFusion --> Group2["Aggregate & Expand Parents"]
        Group2 --> Rerank2["CrossEncoder Rerank (top_n=5)"]
        Rerank2 --> PackHyDE["Pack Context & Citations (HyDE)"]

        PackDirect --> ContextReturn["Return Context & Artifact"]
        PackHyDE --> ContextReturn
        ContextReturn -.->|Context + Citations| StrongBranch
    end

    subgraph Step4 ["4. Grounding & Post-Processing"]
        WeakBranch --> CheckEvidence{"Has RAG Context / Citations?"}
        StrongBranch --> CheckEvidence
        
        CheckEvidence -->|No: Non-RAG/Simple| AutoGrounded["is_grounded: True<br/>score: 1.0, unsupported: []"]
        CheckEvidence -->|Yes: RAG Answer| CitationAlignment["Extract inline citation: chunk_id"]
        
        CitationAlignment --> EvalBranch{"enable_grounding_eval?"}
        EvalBranch -->|False (Default)| FastHeuristic["Heuristic Match: Valid chunk_ids"]
        EvalBranch -->|True| LLMJudge["LLM-as-a-Judge (Weak LLM)"]

        AutoGrounded --> FinalResp["Assemble ChatQueryResponse"]
        FastHeuristic --> FinalResp
        LLMJudge --> FinalResp
        FinalResp --> SaveTurn["SessionManager.record_turn()"]
    end

    SaveTurn --> ClientResponse["Return JSON Response"]
```

---

### Sơ đồ dạng Text / ASCII (Đảm bảo xem được trên mọi trình Preview Markdown)

```text
===================================================================================================
                                      RAG QUERY WORKFLOW
===================================================================================================

[ USER / CLIENT ] 
       │ (POST /api/v1/chat/query)
       ▼
[ FastAPI ChatController ] ──► [ ChatService ]
                                    │
    ┌───────────────────────────────┴──────────────────────────────┐
    ▼                                                              ▼
1. SESSION & CONTEXT ANALYZER                                2. DYNAMIC ROUTER
   • Lấy 4 lượt chat gần nhất từ SessionManager                 • is_simple() regex check
   • needs_context() check:                                        ├── True (Chào hỏi / Persona)
     ├── CONTINUE ──► Rewrite query độc lập                            └── Weak LLM (gpt-4o-mini), No Tools
     └── NEW_TOPIC ──► Giữ nguyên câu hỏi                          └── False (Câu hỏi nghiệp vụ / RAG)
                                                                       └── Strong LLM (gpt-4o) + Tool search
                                                                            │
    ┌───────────────────────────────────────────────────────────────────────┘
    ▼
3. RETRIEVAL TOOL PIPELINE (search_documents)
   │
   ├── [ STAGE 1: DIRECT RETRIEVAL ]
   │     • Hybrid Search (Dense + Sparse Qdrant, k=10)
   │     • Aggregate Chunks by Section / Parent
   │     • Top Parents Expansion (ghép anh em cùng section)
   │     • CrossEncoder Rerank (Qwen3-Reranker-0.6B)
   │     • Gate: top1 >= 0.5 HOẶC mean >= 0.3 ?
   │         ├── ĐẠT ───────────► Đóng gói Context & Citations (Direct branch)
   │         └── CHƯA ĐẠT ──────► [ STAGE 2: HyDE FALLBACK ]
   │                                  • Weak LLM sinh n câu paraphrase
   │                                  • Multi-query Hybrid Search
   │                                  • RRF Fusion (Reciprocal Rank Fusion)
   │                                  • Section Expansion & CrossEncoder Rerank
   │                                  • Đóng gói Context & Citations (HyDE branch)
   │
   └──────► [ Gửi Context + Citations vào Strong LLM để sinh câu trả lời ]
                 │
                 ▼
4. GROUNDING & POST-PROCESSING
   • Có dữ liệu RAG (context/citations) không?
     ├── KHÔNG (Non-RAG / Simple) ──► is_grounded: True, score: 1.0, unsupported_claims: []
     └── CÓ (RAG Answer)          ──► Đối soát nhãn trích dẫn [chunk_id]
                                        ├── ENABLE_GROUNDING_EVAL=false: Heuristic (0 token)
                                        └── ENABLE_GROUNDING_EVAL=true: LLM Judge
   • Lưu lịch sử chat vào SessionManager
   • Trả về JSON ChatQueryResponse
===================================================================================================
```

---

## 2. Chi tiết các thành phần cốt lõi

### A. Pre-processing: Phân tích ngữ cảnh & Đa lượt (`ConversationAnalyzer`)
1. **Lấy lịch sử hội thoại:** Trích xuất 4 message gần nhất từ `SessionManager` theo `session_id`.
2. **Đánh giá chủ đề (`needs_context`):**
   - Sử dụng mô hình `weak_model` (`gpt-4o-mini`) hoặc Heuristic regex `_COREFERENCE_PATTERN` để phát hiện đại từ thay thế (*nó, cái này, cái đó, máy này, tính năng đó...*).
   - Nếu là **`CONTINUE`**: Gọi prompt tóm tắt ngữ cảnh trước + viết lại thành một câu hỏi độc lập (**standalone query**) đầy đủ chủ ngữ và thực thể trước khi tìm kiếm.
   - Nếu là **`NEW_TOPIC`**: Giữ nguyên truy vấn của người dùng.

---

### B. Dynamic Router: Điều hướng thông minh & Tiết kiệm Token
Trước khi gọi mô hình agent, `dynamic_router` kiểm tra độ dài và đặc tính của câu hỏi:
- **`is_simple()`**: Nhận diện các câu chào hỏi, cảm ơn, tạm biệt, câu hỏi danh tính/giới thiệu ngắn (`length < 50`):
  - **Mô hình:** `weak_model` (`gpt-4o-mini`).
  - **Tools:** `[]` (Không truyền schema tool `search_documents`, tiết kiệm hàng trăm prompt token).
  - **Prompt:** `SIMPLE_PROMPT` ngắn gọn.
- **RAG / Domain Queries:**
  - **Mô hình:** `strong_model` (`gpt-4o`).
  - **Tools:** `[search_documents]`.
  - **Prompt:** `SYSTEM_PROMPT` với quy tắc bám sát tài liệu và trích dẫn `[chunk_id]`.

---

### C. Công cụ tìm kiếm 2 tầng (`search_documents` Pipeline)

Quy trình tìm kiếm được tối ưu theo cơ chế **Adaptive / Two-Stage Retrieval**:

1. **Stage 1 (Direct Retrieval):**
   - **Hybrid Search:** Gọi Qdrant kết hợp Dense Vector (Embedding) + Sparse Vector (BM25/Sparse SPLADE) với `direct_k = 10`.
   - **Section / Parent Aggregation:** Gom nhóm các chunk theo section/parent ID và tính điểm trung bình, điểm cao nhất.
   - **Parent Expansion:** Mở rộng ngữ cảnh để lấy thêm các chunk anh em trong cùng một section (đảm bảo tính liền mạch của tài liệu).
   - **Reranker:** Đưa qua `CrossEncoder (Qwen3-Reranker-0.6B)` chấm điểm lại.
   - **Confidence Gate:**
     $$\text{Top 1 score} \ge 0.5 \quad \text{HOẶC} \quad \text{Mean score} \ge 0.3$$
     - Nếu đạt: Trả về kết quả ngay lập tức (tiết kiệm thời gian và lượt gọi HyDE).
2. **Stage 2 (HyDE Fallback - Khi Stage 1 không đủ tự tin):**
   - Dùng `weak_model` sinh ra $N$ câu paraphrase giả định (Hypothetical Document Embeddings).
   - Thực hiện tìm kiếm Hybrid đồng thời trên truy vấn gốc + các câu paraphrases.
   - **RRF (Reciprocal Rank Fusion):** Hợp nhất bảng xếp hạng từ nhiều luồng tìm kiếm:
     $$RRF\_Score(d) = \sum_{q \in Q} \frac{1}{k_{rrf} + \text{rank}(d, q)}$$
   - Thực hiện lại Parent Expansion và Reranking với CrossEncoder để lấy ra $top\_n = 5$ đoạn văn tốt nhất.

---

### D. Grounding & Anti-Hallucination Verification

Sau khi Agent sinh câu trả lời:
1. **Trích xuất Citation:** Trích xuất các nhãn trích dẫn `[chunk_id]` có trong câu trả lời và đối soát với danh sách chunk thực tế được trả về từ retrieval.
2. **Phân nhánh thẩm định:**
   - **Trường hợp phi RAG / Non-RAG:** Nếu không có context tài liệu và câu hỏi là hội thoại thông thường $\rightarrow$ Tự động gán `is_grounded: True`, `grounding_score: 1.0`, `unsupported_claims: []`.
   - **Trường hợp từ chối hợp lệ (`is_grounded_refusal`):** Model trả lời không có thông tin trong tài liệu $\rightarrow$ Coi là grounded (`score: 1.0`).
   - **Kiểm tra Heuristic (Mặc định `ENABLE_GROUNDING_EVAL=false`):** 
     - Có ít nhất 1 `chunk_id` hợp lệ $\rightarrow$ `is_grounded: True`, tính điểm theo số lượng chunk trích dẫn.
     - Thiếu trích dẫn $\rightarrow$ Gán `unsupported_claims: ["Answer lacks inline citation [chunk_id] supporting the claims"]`.
   - **LLM Judge (`ENABLE_GROUNDING_EVAL=true`):** Gửi ngữ cảnh và câu trả lời lên `weak_model` đánh giá bằng JSON.

---

## 3. Cấu hình Model hiện tại

| Vai trò | Model chỉ định | Mục đích sử dụng |
| :--- | :--- | :--- |
| **Weak LLM (`LLM_LV1`)** | `openai:gpt-4o-mini` | Sinh HyDE paraphrases, tóm tắt multiturn, router câu đơn giản |
| **Strong LLM (`LLM_LV2`)** | `openai:gpt-4o` | Agent lập luận chính, gọi tool `search_documents`, trả lời câu hỏi RAG |
| **Embedding Model** | `ollama:qwen3-vl-embedding-2b` | Mã hóa Dense vector lưu trữ trên Qdrant |
| **Reranker Model** | `Qwen/Qwen3-Reranker-0.6B` | Chấm điểm độ tương quan cross-encoder giữa Query và Chunks |

