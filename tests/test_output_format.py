"""Citation phải khớp đúng format data_pipeline đã lưu (output/*.json)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.database.assets import AssetService
from infra.section_store import SectionStore
from model.res.chat_res import ChatQueryResponse
from tools.search_tool import _pack, depth_of

OUTPUT = Path(__file__).resolve().parents[1] / "output"

pytestmark = pytest.mark.skipif(not (OUTPUT / "chunks_metadata.json").exists(),
                                reason="chưa có output/ (chạy data_pipeline.ipynb trước)")


@pytest.fixture(scope="module")
def chunks_meta():
    return json.loads((OUTPUT / "chunks_metadata.json").read_text(encoding="utf-8"))


def _doc(meta):
    return SimpleNamespace(metadata=meta, page_content="body text")


def test_citation_mirror_metadata(chunks_meta):
    assets = AssetService.from_output_dir(str(OUTPUT))
    sections = SectionStore.from_output_dir(str(OUTPUT))
    meta = chunks_meta[0]  # chunk gốc, có images

    context, artifact = _pack([(_doc(meta), 0.87)], "q", [], "direct", (0.87, 0.87),
                               assets, 50, sections)

    cite = artifact["citations"][0]
    # mọi field của metadata đã lưu phải còn nguyên trong citation
    for key, value in meta.items():
        if key in ("images", "tables"):
            continue
        assert cite[key] == value
    assert cite["rerank_score"] == pytest.approx(0.87)
    assert cite["depth"] == 0  # chunk ROOT
    assert meta["source_file"] in context  # nhãn section fallback về source_file

    # ảnh: đúng shape images_metadata (image_id/file/page_no/match)
    assert cite["images"]
    img = cite["images"][0]
    assert set(img) >= {"image_id", "file", "page_no", "match"}
    assert img["image_id"] == meta["images"][0]["image_id"]
    assert img["file"] == meta["images"][0]["file"]


def test_table_rows_no_bom(chunks_meta):
    assets = AssetService.from_output_dir(str(OUTPUT))
    sections = SectionStore.from_output_dir(str(OUTPUT))
    meta = next(m for m in chunks_meta if m.get("tables"))

    _, artifact = _pack([(_doc(meta), 0.5)], "q", [], "direct", (0.5, 0.5),
                         assets, 50, sections)

    tbl = artifact["citations"][0]["tables"][0]
    assert tbl["table_id"] == meta["tables"][0]["table_id"]
    assert tbl["csv"] == meta["tables"][0]["csv"]
    assert tbl["rows"], "bảng phải có rows đọc từ CSV"
    assert not tbl["rows"][0][0].startswith("\ufeff"), "CSV đọc phải bỏ BOM"


def test_depth_from_section_tree(chunks_meta):
    sections = SectionStore.from_output_dir(str(OUTPUT))
    meta = next(m for m in chunks_meta if m["section_path"] != "__ROOT__")
    assert depth_of(_doc(meta), sections) == sections.nodes[meta["section_id"]]["level"]


def test_response_model_accepts_citation(chunks_meta):
    assets = AssetService.from_output_dir(str(OUTPUT))
    sections = SectionStore.from_output_dir(str(OUTPUT))
    meta = chunks_meta[0]
    _, artifact = _pack([(_doc(meta), 0.1)], "q", [], "direct", (0.1, 0.1),
                         assets, 50, sections)

    res = ChatQueryResponse(answer="a", query_norm="q", citations=artifact["citations"])
    assert res.citations[0].chunk_id == meta["chunk_id"]
    assert res.citations[0].images[0].image_id == meta["images"][0]["image_id"]


def test_section_resolve_skips_root(chunks_meta):
    sections = SectionStore.from_output_dir(str(OUTPUT))
    tree = json.loads((OUTPUT / "section_tree.json").read_text(encoding="utf-8"))
    # tên file PDF là title của ROOT -> không được match
    assert sections.resolve(tree["source_file"]) is None
