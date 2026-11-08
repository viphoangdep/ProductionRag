"""Pure helper functions for corpus building, chunking, and geometry."""
from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from pathlib import Path

ROOT_PATH = "__ROOT__"
GLOBAL_ROOT_ID = "root"


def make_doc_slug(pdf_path: Path) -> str:
    """Tạo slug an toàn từ tên file PDF."""
    return re.sub(r"[^A-Za-z0-9_-]+", "", pdf_path.stem.replace(" ", "_"))


def make_chunk_id(doc_slug: str, version: str, index: int) -> str:
    """Tạo mã định danh duy nhất cho từng chunk."""
    return f"{doc_slug}-{version}-{index:04d}"


def point_id_for(chunk_id: str) -> str:
    """Point ID ổn định theo chunk_id (uuid5) -> resume đúng mọi thứ từ file."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def make_section_path(headings: list) -> str:
    """Ghép chuỗi đường dẫn cây tiêu đề (headings)."""
    if not headings:
        return ""
    return " > ".join(headings)


def make_node_id(source_file: str, section_path: str) -> str:
    """Sinh ID node section duy nhất dựa trên md5 của source_file + section_path."""
    raw = f"{source_file}::{section_path}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"parent_{digest}"


def normalize_headings(headings) -> list:
    """Làm sạch và loại bỏ heading rỗng hoặc None."""
    if not headings:
        return []
    result = []
    for heading in headings:
        if heading is None:
            continue
        value = heading.strip() if isinstance(heading, str) else str(heading).strip()
        if value:
            result.append(value)
    return result


def heading_info(source_file: str, headings: list) -> tuple:
    """Trả về (section_path, section_id, parent_id, parent_title, ancestor_ids, depth)."""
    if not headings:
        root = make_node_id(source_file, ROOT_PATH)
        return ROOT_PATH, root, None, "__ROOT__", [], 0
    section_path = make_section_path(headings)
    section_id = make_node_id(source_file, section_path)
    if len(headings) == 1:
        parent_id, parent_title = make_node_id(source_file, ROOT_PATH), "__ROOT__"
    else:
        parent_path = make_section_path(headings[:-1])
        parent_id = make_node_id(source_file, parent_path)
        parent_title = headings[-2]
    ancestor_ids = [
        make_node_id(source_file, make_section_path(headings[:k]))
        for k in range(1, len(headings))
    ]
    return section_path, section_id, parent_id, parent_title, ancestor_ids, len(headings)


def get_chunk_pages(chunk) -> list:
    """Trích xuất danh sách số trang thuộc chunk."""
    pages = set()
    for item in getattr(chunk.meta, "doc_items", []):
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(page_no)
    return sorted(pages)


def get_chunk_page_boxes(chunk) -> dict:
    """Tính toán bounding box tổng hợp của chunk trên từng trang."""
    page_boxes = defaultdict(list)
    for item in getattr(chunk.meta, "doc_items", []):
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is None:
                continue
            bbox = getattr(prov, "bbox", None)
            if bbox is not None:
                page_boxes[page_no].append(bbox)
    out = {}
    for page_no, boxes in page_boxes.items():
        out[page_no] = {
            "left": min(b.l for b in boxes),
            "right": max(b.r for b in boxes),
            "bottom": min(b.b for b in boxes),
            "top": max(b.t for b in boxes),
        }
    return out


def vertical_distance(picture_bbox, chunk_box) -> float:
    """Khoảng cách theo phương thẳng đứng giữa ảnh và chunk."""
    pt, pb = picture_bbox.t, picture_bbox.b
    ct, cb = chunk_box["top"], chunk_box["bottom"]
    if pb <= ct and pt >= cb:
        return 0.0
    if pb > ct:
        return pb - ct
    return cb - pt


def chunk_body(chunks_dir: Path, chunk_id: str) -> str:
    """Đọc nội dung văn bản markdown của chunk, bỏ qua frontmatter YAML."""
    raw = (chunks_dir / f"{chunk_id}.md").read_text(encoding="utf-8")
    parts = raw.split("---\n")
    if len(parts) >= 3:
        return "---\n".join(parts[2:]).strip()
    return raw.strip()

