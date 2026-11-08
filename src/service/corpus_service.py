"""CorpusService: Điều phối nghiệp vụ upload tài liệu, build corpus và theo dõi background tasks."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from config.settings import settings
from infra.corpus_pipeline import (
    build_chunker,
    build_converter,
    build_s3,
    embed_with_retry,
    ensure_collection,
    get_existing_ids,
    merge_outputs,
    process_one_pdf,
)
from service.task_tracker import TaskTracker
from util.corpus_util import (
    GLOBAL_ROOT_ID,
    ROOT_PATH,
    chunk_body,
    get_chunk_page_boxes,
    get_chunk_pages,
    heading_info,
    make_chunk_id,
    make_doc_slug,
    make_node_id,
    make_section_path,
    normalize_headings,
    point_id_for,
    vertical_distance,
)

log = logging.getLogger("ai-service")


class CorpusService:
    """Service điều phối nghiệp vụ xây dựng và cập nhật dữ liệu vector store từ tài liệu PDF."""

    def __init__(self, cfg=None, task_tracker: TaskTracker | None = None):
        self.cfg = cfg or settings
        self.tracker = task_tracker or TaskTracker()

    # --- Task Tracker delegates (bảo toàn tương thích với các unit tests) ---
    def _create_task(self, total_files: int = 0) -> str:
        return self.tracker.create_task(total_files=total_files)

    def _update_task_progress(self, task_id: str, processed_files: int, current_file: str | None = None):
        self.tracker.update_progress(task_id, processed_files=processed_files, current_file=current_file)

    def _complete_task(self, task_id: str, result: dict[str, Any]):
        self.tracker.complete_task(task_id, result=result)

    def _fail_task(self, task_id: str, error: str):
        self.tracker.fail_task(task_id, error=error)

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        return self.tracker.get_task(task_id)

    # --- Document Upload & File Handling ---
    def save_uploaded_files(self, files: list[Any]) -> list[Path]:
        """Lưu danh sách file PDF được upload từ client vào thư mục docs_dir."""
        docs_path = self.cfg.docs_path
        docs_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for file in files:
            filename = Path(getattr(file, "filename", "unnamed.pdf")).name
            target = docs_path / filename
            content = file.file.read() if hasattr(file, "file") else getattr(file, "content", b"")
            target.write_bytes(content)
            saved_paths.append(target)
        return saved_paths

    # --- Background Execution ---
    def start_build_task(
        self,
        pdf_paths: list[Path] | None = None,
        recreate_collection: bool = False,
        skip_s3: bool = False,
        only: list[str] | None = None,
        version: str = "v1",
    ) -> str:
        """Khởi động tác vụ build corpus bất đồng bộ chạy nền trong background thread."""
        docs_path = self.cfg.docs_path
        if pdf_paths is None:
            pdf_paths = sorted(p for p in docs_path.rglob("*.pdf") if p.is_file())
        if only:
            want = set(only)
            pdf_paths = [p for p in pdf_paths if make_doc_slug(p) in want]

        task_id = self.tracker.create_task(total_files=len(pdf_paths))

        def _worker():
            try:
                self.tracker.update_progress(task_id, processed_files=0)
                result = self.build_corpus_sync(
                    pdf_paths=pdf_paths,
                    output_dir=self.cfg.output_path,
                    recreate_collection=recreate_collection,
                    skip_s3=skip_s3,
                    version=version,
                    task_id=task_id,
                )
                self.tracker.complete_task(task_id, result)
            except Exception as e:
                log.exception("Background corpus build task failed: %s", e)
                self.tracker.fail_task(task_id, str(e))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return task_id

    # --- Synchronous Execution (cho CLI hoặc worker) ---
    def build_corpus_sync(
        self,
        docs_dir: Path | str | None = None,
        pdf_paths: list[Path] | None = None,
        output_dir: Path | str | None = None,
        recreate_collection: bool = False,
        skip_s3: bool = False,
        only: list[str] | None = None,
        version: str = "v1",
        task_id: str | None = None,
    ) -> dict:
        """Thực thi đồng bộ toàn bộ luồng pipeline build corpus."""
        output_path = Path(output_dir) if output_dir else self.cfg.output_path
        if pdf_paths is None:
            dir_path = Path(docs_dir) if docs_dir else self.cfg.docs_path
            pdf_paths = sorted(p for p in dir_path.rglob("*.pdf") if p.is_file())
        if only:
            want = set(only)
            pdf_paths = [p for p in pdf_paths if make_doc_slug(p) in want]

        if not pdf_paths:
            log.warning("No PDF files found to process.")
            return {}

        chunks_dir = output_path / "chunks"
        tables_dir = output_path / "tables"
        (output_path / "per_doc").mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        from langchain_ollama import OllamaEmbeddings
        from langchain_qdrant import FastEmbedSparse
        from qdrant_client import QdrantClient

        client = QdrantClient(url=self.cfg.qdrant_url)
        embeddings = OllamaEmbeddings(model=self.cfg.embed_model, base_url="http://localhost:11434")
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        ensure_collection(client, self.cfg.collection, "dense", "sparse",
                          embeddings, recreate_collection)
        existing = get_existing_ids(client, self.cfg.collection)

        s3, bucket, region = (None, None, None) if (skip_s3 or self.cfg.skip_s3) else build_s3(self.cfg)
        chunker, hf_tokenizer = build_chunker(self.cfg.embed_model_id, self.cfg.chunk_max_tokens)

        ctx = {
            "version": version, "output_dir": output_path,
            "chunks_dir": chunks_dir, "tables_dir": tables_dir,
            "converter": build_converter(),
            "chunker": chunker, "hf_tokenizer": hf_tokenizer,
            "client": client, "collection": self.cfg.collection,
            "embeddings": embeddings, "sparse_embeddings": sparse_embeddings,
            "existing": existing, "text_batch": self.cfg.text_batch,
            "vector_name": "dense", "sparse_name": "sparse",
            "s3": s3, "bucket": bucket, "region": region,
            "skip_s3": skip_s3 or self.cfg.skip_s3 or (s3 is None),
        }

        results = []
        for i, pdf in enumerate(pdf_paths, start=1):
            if task_id:
                self.tracker.update_progress(task_id, processed_files=i - 1, current_file=pdf.name)
            try:
                results.append(process_one_pdf(ctx, pdf))
            except Exception as e:
                log.exception("FAILED processing %s: %s", pdf.name, e)
            if task_id:
                self.tracker.update_progress(task_id, processed_files=i, current_file=None)

        manifest = merge_outputs(output_path, results)
        return manifest

    # --- Manifest & Stats ---
    def get_manifest(self) -> dict[str, Any]:
        """Lấy thông tin manifest tài liệu hiện có và số điểm vector trong Qdrant."""
        manifest_file = self.cfg.output_path / "manifest.json"
        manifest_data = {}
        if manifest_file.exists():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Could not read manifest.json: %s", e)

        points_count = None
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=self.cfg.qdrant_url)
            if client.collection_exists(self.cfg.collection):
                info = client.get_collection(self.cfg.collection)
                points_count = info.points_count
        except Exception as e:
            log.debug("Could not get collection stats from Qdrant: %s", e)

        return {
            "total_docs": len(manifest_data),
            "points_count": points_count,
            "manifest": manifest_data,
        }
