#!/usr/bin/env python3
"""CLI build RAG corpus từ nhiều PDF trong docs/.

Ủy quyền toàn bộ logic nghiệp vụ cho CorpusService trong src/service/corpus_service.py.
Chạy: python scripts/build_corpus.py [--docs docs] [--output output] [--recreate-collection] [--skip-s3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Đảm bảo src/ nằm trong sys.path khi chạy trực tiếp từ CLI
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config.settings import settings
from service.corpus_service import (
    ROOT_PATH,
    GLOBAL_ROOT_ID,
    CorpusService,
    make_doc_slug,
    make_chunk_id,
    point_id_for,
    make_section_path,
    make_node_id,
    normalize_headings,
    heading_info,
    get_chunk_pages,
    get_chunk_page_boxes,
    vertical_distance,
    chunk_body,
    embed_with_retry,
    build_converter,
    build_chunker,
    build_s3,
    ensure_collection,
    get_existing_ids,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build RAG corpus từ nhiều PDF (streaming, giới hạn RAM).")
    ap.add_argument("--docs", default=settings.docs_dir, help="Thư mục chứa file PDF")
    ap.add_argument("--output", default=settings.output_dir, help="Thư mục xuất metadata và chunks")
    ap.add_argument("--version", default="v1", help="Phiên bản dữ liệu")
    ap.add_argument("--collection", default=settings.collection, help="Tên Qdrant collection")
    ap.add_argument("--qdrant-url", default=settings.qdrant_url, help="URL máy chủ Qdrant")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="URL máy chủ Ollama")
    ap.add_argument("--embed-model", default=settings.embed_model, help="Model dense embeddings")
    ap.add_argument("--embed-model-id", default=settings.embed_model_id, help="Tokenizer model ID cho chunker")
    ap.add_argument("--max-tokens", type=int, default=settings.chunk_max_tokens, help="Số token tối đa mỗi chunk")
    ap.add_argument("--text-batch", type=int, default=settings.text_batch, help="Kích thước batch embed")
    ap.add_argument("--recreate-collection", action="store_true", help="Xóa và tạo mới collection")
    ap.add_argument("--skip-s3", action="store_true", help="Bỏ qua upload S3")
    ap.add_argument("--only", default="", help="Chỉ chạy các slug cách nhau dấu phẩy")
    args = ap.parse_args(argv)

    # Cập nhật cài đặt theo tham số dòng lệnh nếu khác mặc định
    settings.docs_dir = args.docs
    settings.output_dir = args.output
    settings.collection = args.collection
    settings.qdrant_url = args.qdrant_url
    settings.embed_model = args.embed_model
    settings.embed_model_id = args.embed_model_id
    settings.chunk_max_tokens = args.max_tokens
    settings.text_batch = args.text_batch
    settings.skip_s3 = args.skip_s3

    only_slugs = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None

    service = CorpusService(cfg=settings)
    manifest = service.build_corpus_sync(
        docs_dir=args.docs,
        output_dir=args.output,
        recreate_collection=args.recreate_collection,
        skip_s3=args.skip_s3,
        only=only_slugs,
        version=args.version,
    )
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
