from pydantic import BaseModel


class TableRef(BaseModel):
    """Shape đúng như tables_metadata.json + rows đọc từ CSV."""

    table_id: str
    page_no: int | None = None
    csv: str | None = None
    html: str | None = None
    doc_item_ref: str | None = None
    rows: list[list[str]] = []


class ImageRef(BaseModel):
    """Shape đúng như images_metadata.json (image_id là id tra ảnh qua API)."""

    image_id: str
    file: str | None = None
    page_no: int | None = None
    match: str | None = None  # "exact" | "bbox" - cách ảnh gắn vào chunk
    doc_item_ref: str | None = None


class Citation(BaseModel):
    """Mirror metadata chunk đã lưu (chunks_metadata.json) + điểm rerank."""

    chunk_id: str
    chunk_index: int | None = None
    source_file: str | None = None
    parent_id: str | None = None
    parent_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    ancestor_ids: list[str] = []
    depth: int = 0
    page_nos: list[int] = []
    rerank_score: float
    tables: list[TableRef] = []
    images: list[ImageRef] = []
    is_cited: bool = False


class ChatQueryResponse(BaseModel):
    session_id: str = ""
    answer: str
    query_norm: str
    standalone_query: str | None = None
    is_new_topic: bool = True
    is_grounded: bool = True
    grounding_score: float | None = None
    cited_chunk_ids: list[str] = []
    unsupported_claims: list[str] = []
    paraphrases: list[str] = []
    citations: list[Citation] = []
