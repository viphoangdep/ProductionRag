"""Corpus ingestion pipeline: Docling PDF parsing, chunking, S3 upload, and Qdrant batch upsert."""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from config.settings import settings
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
    normalize_headings,
    point_id_for,
    vertical_distance,
)

log = logging.getLogger("ai-service")


def embed_with_retry(embeddings, batch: list, attempts: int = 5) -> list:
    """Retry + backoff cho Ollama/Embedding provider."""
    delay = 5
    for a in range(attempts):
        try:
            return embeddings.embed_documents(batch)
        except Exception as e:
            if a == attempts - 1:
                raise
            log.warning("embed retry %d/%d: %s — sleep %ds", a + 1, attempts, type(e).__name__, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def build_converter():
    """Khởi tạo DocumentConverter của Docling với OCR và table structure."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = False
    pipeline_options.images_scale = 2.0
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def build_chunker(embed_model_id: str, max_tokens: int):
    """Khởi tạo HybridChunker và HuggingFace tokenizer."""
    from transformers import AutoTokenizer
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens)
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    return chunker, hf_tokenizer


def build_s3(cfg=None):
    """Khởi tạo AWS S3 client từ cấu hình hoặc biến môi trường."""
    import boto3

    cfg = cfg or settings
    access = cfg.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    secret = cfg.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    region = cfg.aws_region or os.getenv("AWS_REGION")
    bucket = cfg.aws_s3_bucket or os.getenv("AWS_S3_BUCKET")
    if not (access and secret and bucket):
        return None, None, None
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
    )
    return s3, bucket, region


def ensure_collection(client, collection: str, vector_name: str, sparse_name: str,
                      embeddings, recreate: bool):
    """Đảm bảo Qdrant collection tồn tại và có đủ payload indexes."""
    from qdrant_client.models import Distance, VectorParams, SparseVectorParams

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
        log.info("deleted collection '%s'", collection)
    if not client.collection_exists(collection):
        dim = len(embeddings.embed_query("dimension test"))
        client.create_collection(
            collection_name=collection,
            vectors_config={vector_name: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={sparse_name: SparseVectorParams()},
        )
        log.info("created collection '%s' dense=%d+sparse=BM25", collection, dim)
    for key, schema in [("metadata.section_id", "keyword"),
                        ("metadata.ancestor_ids", "keyword"),
                        ("metadata.page_nos", "integer")]:
        try:
            client.create_payload_index(collection_name=collection,
                                        field_name=key, field_schema=schema)
            log.info("index ok: %s", key)
        except Exception as e:
            log.debug("index skip %s: %s", key, e)


def get_existing_ids(client, collection: str) -> set:
    """Lấy danh sách các point_id đã tồn tại trong collection để skip."""
    existing = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_vectors=False,
            with_payload=False,
        )
        existing.update(str(p.id) for p in points)
        if offset is None:
            break
    return existing


def process_one_pdf(ctx: dict, pdf_path: Path) -> dict:
    """Xử lý đơn lẻ 1 file PDF và streaming upsert vào Qdrant."""
    doc_slug = make_doc_slug(pdf_path)
    version = ctx["version"]
    source_file = pdf_path.name
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    out_doc = ctx["output_dir"] / "per_doc" / doc_slug
    out_doc.mkdir(parents=True, exist_ok=True)
    log.info("Processing %s -> %s", pdf_path.name, doc_slug)

    doc = ctx["converter"].convert(pdf_path).document

    from docling_core.types.doc import PictureItem, TableItem

    # ---- images
    image_metadata, image_blobs = [], {}
    for element, level in doc.iterate_items():
        if not isinstance(element, PictureItem):
            continue
        n = len(image_metadata) + 1
        image_id = f"{doc_slug}-img{n:04d}"
        try:
            image = element.get_image(doc)
            buf = io.BytesIO()
            image.save(buf, "PNG")
            image_blobs[image_id] = buf.getvalue()
            page_no = element.prov[0].page_no if element.prov else None
            image_metadata.append({
                "image_id": image_id, "file": None, "s3_key": None, "s3_url": None,
                "source_file": source_file, "created_at": created_at,
                "doc_item_ref": str(element.self_ref), "page_no": page_no,
                "width": image.width, "height": image.height, "level": level,
            })
        except Exception as e:
            log.warning("failed image %d: %s", n, e)

    # ---- tables
    table_metadata, table_html_blobs, table_csv = [], {}, {}
    for element, level in doc.iterate_items():
        if not isinstance(element, TableItem):
            continue
        n = len(table_metadata) + 1
        table_id = f"{doc_slug}-tbl{n:04d}"
        csv_path = ctx["tables_dir"] / f"{table_id}.csv"
        try:
            doc_item_ref = str(element.self_ref)
        except Exception:
            doc_item_ref = None
        page_no = element.prov[0].page_no if element.prov else None
        df = None
        try:
            df = element.export_to_dataframe(doc=doc)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            table_csv[table_id] = csv_path.read_text(encoding="utf-8-sig")
        except Exception as e:
            log.warning("table %s CSV failed: %s", table_id, e)
        try:
            table_html_blobs[table_id] = element.export_to_html(doc=doc)
        except Exception as e:
            log.warning("table %s HTML failed: %s", table_id, e)
        table_metadata.append({
            "table_id": table_id, "source_file": source_file, "created_at": created_at,
            "doc_item_ref": doc_item_ref, "page_no": page_no,
            "csv": csv_path.relative_to(ctx["output_dir"]).as_posix() if csv_path.exists() else None,
            "html": None, "s3_key": None, "s3_url": None,
            "rows": len(df) if df is not None else None,
            "columns": len(df.columns) if df is not None else None,
            "column_names": [str(c) for c in df.columns] if df is not None else [],
            "level": level,
        })

    # ---- chunk
    chunker, hf_tokenizer = ctx["chunker"], ctx["hf_tokenizer"]
    chunks = list(chunker.chunk(dl_doc=doc))

    image_map = {m["doc_item_ref"]: m for m in image_metadata if m.get("doc_item_ref")}
    table_map = {m["doc_item_ref"]: m for m in table_metadata if m.get("doc_item_ref")}
    chunk_boxes_list = [get_chunk_page_boxes(c) for c in chunks]

    pictures_by_page = defaultdict(list)
    for element, _level in doc.iterate_items():
        if not isinstance(element, PictureItem):
            continue
        ref = str(element.self_ref)
        for prov in element.prov or []:
            page_no, bbox = getattr(prov, "page_no", None), getattr(prov, "bbox", None)
            if page_no is None or bbox is None:
                continue
            pictures_by_page[page_no].append({"picture_ref": ref, "bbox": bbox})

    picture_to_chunk = {}
    for page, pics in pictures_by_page.items():
        for pic in pics:
            best, best_d = None, float("inf")
            for idx, boxes in enumerate(chunk_boxes_list):
                if page not in boxes:
                    continue
                d = vertical_distance(pic["bbox"], boxes[page])
                if d < best_d:
                    best_d, best = d, idx
            if best is not None:
                prev = picture_to_chunk.get(pic["picture_ref"])
                if prev is None or best_d < prev["distance"]:
                    picture_to_chunk[pic["picture_ref"]] = {"chunk_index": best, "distance": best_d}
    bbox_refs_by_chunk = defaultdict(list)
    for ref, mapping in picture_to_chunk.items():
        bbox_refs_by_chunk[mapping["chunk_index"]].append(ref)
    del pictures_by_page, picture_to_chunk
    gc.collect()

    # ---- build chunks + tree
    raw_chunks, last_headings = [], []
    for index, chunk in enumerate(chunks):
        chunk_id = make_chunk_id(doc_slug, version, index)
        text = chunker.contextualize(chunk=chunk)
        num_tokens = len(hf_tokenizer.encode(text, add_special_tokens=False))
        raw_headings = normalize_headings(getattr(chunk.meta, "headings", None))
        current_headings = raw_headings or last_headings
        if raw_headings:
            last_headings = raw_headings
        sp, sid, pid, ptitle, anc, depth = heading_info(source_file, current_headings)
        raw_chunks.append({
            "index": index, "chunk": chunk, "chunk_id": chunk_id,
            "text": text, "num_tokens": num_tokens, "headings": current_headings,
            "section_path": sp, "section_id": sid,
            "section_title": current_headings[-1] if current_headings else "__ROOT__",
            "parent_id": pid, "parent_title": ptitle,
            "ancestor_ids": anc, "ancestor_titles": current_headings[:-1], "depth": depth,
            "doc_items": [{"ref": str(it.self_ref), "label": str(it.label)}
                          for it in getattr(chunk.meta, "doc_items", [])],
            "page_nos": get_chunk_pages(chunk),
        })
    del chunks, chunk_boxes_list
    gc.collect()

    parent_groups = defaultdict(list)
    section_nodes = {}
    for item in raw_chunks:
        parent_groups[item["section_id"]].append(item["index"])
        sid = item["section_id"]
        if sid not in section_nodes:
            section_nodes[sid] = {
                "node_id": sid, "title": item["section_title"], "level": item["depth"],
                "parent_id": item["parent_id"], "section_path": item["section_path"],
                "ancestor_ids": item["ancestor_ids"], "chunk_ids": [],
            }
        section_nodes[sid]["chunk_ids"].append(item["chunk_id"])
    for sid, node in section_nodes.items():
        pid = node["parent_id"]
        if pid and pid in section_nodes:
            section_nodes[pid].setdefault("child_node_ids", []).append(sid)
    for node in section_nodes.values():
        node.setdefault("child_node_ids", [])
    doc_root = make_node_id(source_file, ROOT_PATH)
    section_nodes[doc_root] = {
        "node_id": doc_root, "title": source_file, "level": 0, "parent_id": GLOBAL_ROOT_ID,
        "section_path": ROOT_PATH, "ancestor_ids": [], "chunk_ids": [],
        "child_node_ids": [sid for sid, n in section_nodes.items()
                           if n.get("parent_id") == doc_root],
    }

    chunk_metadata = []
    for item in raw_chunks:
        sib = parent_groups[item["section_id"]]
        images, tables, exact_refs = [], [], set()
        for doc_item in getattr(item["chunk"].meta, "doc_items", []):
            ref = str(doc_item.self_ref)
            if ref in image_map and ref not in exact_refs:
                m = image_map[ref]
                images.append({"image_id": m["image_id"], "file": m["file"],
                               "page_no": m["page_no"], "doc_item_ref": m["doc_item_ref"],
                               "match": "exact"})
                exact_refs.add(ref)
            if ref in table_map:
                m = table_map[ref]
                tables.append({"table_id": m["table_id"], "csv": m["csv"], "html": m["html"],
                               "page_no": m["page_no"], "doc_item_ref": m["doc_item_ref"]})
        for ref in bbox_refs_by_chunk.get(item["index"], []):
            if ref in exact_refs or ref not in image_map:
                continue
            m = image_map[ref]
            images.append({"image_id": m["image_id"], "file": m["file"],
                           "page_no": m["page_no"], "doc_item_ref": m["doc_item_ref"],
                           "match": "bbox"})
        chunk_metadata.append({
            "chunk_id": item["chunk_id"], "chunk_index": item["index"],
            "source_file": source_file, "num_tokens": item["num_tokens"],
            "section_id": item["section_id"], "section_title": item["section_title"],
            "parent_id": item["parent_id"], "parent_title": item["parent_title"],
            "ancestor_ids": item["ancestor_ids"], "ancestor_titles": item["ancestor_titles"],
            "depth": item["depth"], "section_path": item["section_path"],
            "headings": item["headings"],
            "parent_chunk_indices": sib,
            "parent_chunk_ids": [make_chunk_id(doc_slug, version, i) for i in sib],
            "doc_items": item["doc_items"], "page_nos": item["page_nos"],
            "images": images, "tables": tables,
            "file": f"chunks/{item['chunk_id']}.md", "created_at": created_at,
        })
        (ctx["chunks_dir"] / f"{item['chunk_id']}.md").write_text(
            f"---\nchunk_id: \"{item['chunk_id']}\"\nsection_path: "
            f"\"{item['section_path'].replace(chr(34), chr(92) + chr(34))}\"\n---\n\n"
            + item["text"] + "\n", encoding="utf-8")

    for item in raw_chunks:
        item.pop("chunk", None)
    del bbox_refs_by_chunk
    gc.collect()

    # ---- S3
    s3, bucket, region = ctx["s3"], ctx["bucket"], ctx["region"]
    prefix = f"{doc_slug}/{version}/"
    if s3 is None or not bucket or ctx["skip_s3"]:
        log.info("S3 upload skipped.")
    else:
        def _url(key):
            return f"https://{bucket}.s3.{region}.amazonaws.com/{key}" if region else None

        img_by_id = {m["image_id"]: m for m in image_metadata}
        tbl_by_id = {m["table_id"]: m for m in table_metadata}
        for meta in chunk_metadata:
            cid = meta["chunk_id"]
            for seq, img in enumerate(meta.get("images", []), start=1):
                data = image_blobs.get(img["image_id"])
                if not data:
                    continue
                key = f"{prefix}images/{cid}-img{seq:02d}.png"
                try:
                    s3.upload_fileobj(io.BytesIO(data), bucket, key,
                                      ExtraArgs={"ContentType": "image/png"})
                    img_by_id[img["image_id"]].update(s3_key=key, s3_url=_url(key))
                    img.update(s3_key=key, s3_url=_url(key))
                except Exception as e:
                    log.warning("S3 upload failed %s: %s", key, e)
            for seq, tbl in enumerate(meta.get("tables", []), start=1):
                html_text = table_html_blobs.get(tbl["table_id"])
                if not html_text:
                    continue
                key = f"{prefix}tables/{cid}-tbl{seq:02d}.html"
                try:
                    s3.put_object(Bucket=bucket, Key=key,
                                  Body=html_text.encode("utf-8"),
                                  ContentType="text/html; charset=utf-8")
                    tbl_by_id[tbl["table_id"]].update(s3_key=key, s3_url=_url(key))
                    tbl.update(s3_key=key, s3_url=_url(key))
                except Exception as e:
                    log.warning("S3 upload failed %s: %s", key, e)

    image_blobs.clear()
    table_html_blobs.clear()
    gc.collect()

    # ---- metadata files
    (out_doc / "chunks_metadata.json").write_text(
        json.dumps(chunk_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_doc / "images_metadata.json").write_text(
        json.dumps(image_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_doc / "tables_metadata.json").write_text(
        json.dumps(table_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_doc / "section_nodes.json").write_text(
        json.dumps({"root_id": doc_root, "source_file": source_file,
                    "nodes": section_nodes}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ---- Qdrant streaming upsert
    from qdrant_client.models import PointStruct, SparseVector

    client = ctx["client"]
    todo = [m for m in chunk_metadata if point_id_for(m["chunk_id"]) not in ctx["existing"]]
    tb = ctx["text_batch"]
    for start in range(0, len(todo), tb):
        batch_meta = todo[start:start + tb]
        inputs = []
        for m in batch_meta:
            extra = ""
            for t in m.get("tables", []):
                if t["table_id"] in table_csv:
                    extra += f"\n\n[TABLE {t['table_id']} page {t.get('page_no')}]\n{table_csv[t['table_id']]}"
            inputs.append(chunk_body(ctx["chunks_dir"], m["chunk_id"]) + extra)
        dense = embed_with_retry(ctx["embeddings"], inputs)
        sparse = ctx["sparse_embeddings"].embed_documents(inputs)
        points = []
        for m, body, dv, sv in zip(batch_meta, inputs, dense, sparse):
            pages = sorted({i.get("page_no") for i in (m.get("images", []) + m.get("tables", []))
                            if i.get("page_no") is not None})
            points.append({
                "id": point_id_for(m["chunk_id"]),
                "vector": {ctx["vector_name"]: dv,
                           ctx["sparse_name"]: SparseVector(indices=list(sv.indices),
                                                            values=list(sv.values))},
                "payload": {
                    "page_content": body,
                    "metadata": {
                        "type": "chunk", "chunk_id": m["chunk_id"],
                        "chunk_index": m["chunk_index"],
                        "section_id": m["section_id"], "section_title": m["section_title"],
                        "parent_id": m["parent_id"], "parent_title": m["parent_title"],
                        "ancestor_ids": m["ancestor_ids"],
                        "ancestor_titles": m["ancestor_titles"],
                        "depth": m["depth"], "headings": m["headings"],
                        "section_path": m["section_path"],
                        "source_file": source_file, "num_tokens": m["num_tokens"],
                        "page_nos": m["page_nos"] or pages,
                        "images": m["images"], "tables": m["tables"],
                        "created_at": m["created_at"], "file": m["file"],
                    },
                },
            })
        client.upsert(collection_name=ctx["collection"],
                      points=[PointStruct(**p) for p in points])
        for m in batch_meta:
            ctx["existing"].add(point_id_for(m["chunk_id"]))
        del inputs, dense, sparse, points, batch_meta
        gc.collect()
        time.sleep(1)

    del doc, image_metadata, table_metadata, table_csv, raw_chunks, chunk_metadata
    del section_nodes, parent_groups, todo
    gc.collect()
    return {"slug": doc_slug, "source_file": source_file, "version": version,
            "created_at": created_at}


def merge_outputs(output_dir: Path, results: list) -> dict:
    """Tổng hợp kết quả per_doc thành manifest và file gộp."""
    chunks, images, tables, nodes = [], [], [], {GLOBAL_ROOT_ID: {
        "node_id": GLOBAL_ROOT_ID, "title": "__ROOT__", "level": -1,
        "parent_id": None, "section_path": ROOT_PATH, "ancestor_ids": [],
        "chunk_ids": [], "child_node_ids": []}}
    roots = {}
    for r in results:
        d = output_dir / "per_doc" / r["slug"]
        chunks.extend(json.loads((d / "chunks_metadata.json").read_text(encoding="utf-8")))
        images.extend(json.loads((d / "images_metadata.json").read_text(encoding="utf-8")))
        tables.extend(json.loads((d / "tables_metadata.json").read_text(encoding="utf-8")))
        tree = json.loads((d / "section_nodes.json").read_text(encoding="utf-8"))
        roots[r["slug"]] = tree["root_id"]
        nodes[GLOBAL_ROOT_ID]["child_node_ids"].append(tree["root_id"])
        for nid, n in tree["nodes"].items():
            n.pop("chunk_indices", None)
            if nid == tree["root_id"]:
                n["parent_id"] = GLOBAL_ROOT_ID
            nodes[nid] = n
    (output_dir / "chunks_metadata.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "images_metadata.json").write_text(
        json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "tables_metadata.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "section_tree.json").write_text(
        json.dumps({"root_id": GLOBAL_ROOT_ID, "roots": roots,
                    "nodes": nodes}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    manifest = {r["slug"]: r for r in results}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("merged: %d chunks, %d nodes, %d docs", len(chunks), len(nodes), len(results))
    return manifest

