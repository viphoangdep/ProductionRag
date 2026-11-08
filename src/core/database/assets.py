"""Asset resolver: tables inline JSON, images as local files under output/."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from core.exception.errors import AppError, ErrorCode
from core.interfaces.asset_store import BaseAssetStore


class AssetService(BaseAssetStore):
    def __init__(self, output_dir: Path, image_index: dict | None = None,
                 table_rows: dict | None = None, table_page: dict | None = None):
        self._out = output_dir
        self.image_index = image_index or {}
        self.table_rows = table_rows or {}
        self.table_page = table_page or {}

    # ponytail: in-memory index; reload from images/tables_metadata.json at startup if corpus grows
    @classmethod
    def from_output_dir(cls, output_dir: str) -> "AssetService":
        out = Path(output_dir)
        image_index: dict = {}
        try:
            for it in json.loads((out / "images_metadata.json").read_text(encoding="utf-8")):
                if it.get("image_id") and it.get("file"):
                    image_index[it["image_id"]] = it["file"]
        except FileNotFoundError:
            pass
        table_rows: dict = {}
        table_page: dict = {}
        try:
            for tb in json.loads((out / "tables_metadata.json").read_text(encoding="utf-8")):
                tid = tb.get("table_id")
                if not tid:
                    continue
                table_page[tid] = tb.get("page_no")
                csv_rel = tb.get("csv")
                if csv_rel:
                    p = out / csv_rel
                    if p.exists():
                        # pipeline ghi CSV bằng utf-8-sig -> đọc cũng phải utf-8-sig, nếu không
                        # ô header đầu sẽ có BOM lọt vào response.
                        with p.open(encoding="utf-8-sig") as f:
                            table_rows[tid] = list(csv.reader(f))
        except FileNotFoundError:
            pass
        return cls(out, image_index, table_rows, table_page)

    def file_for(self, image_id: str) -> str:
        return self.image_index[image_id]  # KeyError if unknown -> 404

    def resolve_for_chunk(self, metadata: dict, max_rows: int = 50) -> dict:
        """Trả về đúng shape đã lưu (images/tables_metadata) + rows của bảng.

        Pipeline lưu: images[{image_id,file,page_no,doc_item_ref,match}],
        tables[{table_id,csv,html,page_no,doc_item_ref}]. Giữ nguyên field đó,
        chỉ bổ sung rows (bảng) từ CSV và file (ảnh) nếu thiếu.
        """
        tables = []
        for t in metadata.get("tables") or []:
            tid = t.get("table_id")
            if not tid:
                continue
            item = {"table_id": tid,
                    "page_no": t.get("page_no", self.table_page.get(tid)),
                    "csv": t.get("csv"), "html": t.get("html"),
                    "doc_item_ref": t.get("doc_item_ref")}
            rows = self.table_rows.get(tid)
            if rows is not None:
                item["rows"] = rows[:max_rows]
            tables.append(item)
        images = []
        for i in metadata.get("images") or []:
            image_id = i.get("image_id") or i.get("asset_id")
            if not image_id:
                continue
            images.append({"image_id": image_id,
                           "file": i.get("file") or self.image_index.get(image_id),
                           "page_no": i.get("page_no"),
                           "match": i.get("match"),
                           "doc_item_ref": i.get("doc_item_ref")})
        return {"tables": tables, "images": images}

    def image_bytes(self, image_id: str) -> tuple[bytes, str]:
        try:
            data = (self._out / self.file_for(image_id)).read_bytes()
        except (KeyError, FileNotFoundError):
            raise AppError(ErrorCode.NOT_FOUND, "image not found") from None
        return data, "image/png"
