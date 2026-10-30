"""Section tree: ten chuong/section -> node_id + subtree filter (doc output/section_tree.json).

Agent hoi 'o chuong 3...' -> search_tool resolve ten -> Qdrant filter giu ca subtree.
Thieu file (corpus cu) -> store rong, resolve None, query chay khong filter.
"""
from __future__ import annotations


class SectionStore:
    def __init__(self, nodes: dict | None = None, root_id: str | None = None):
        self.nodes = nodes or {}
        self.root_id = root_id

    @classmethod
    def from_output_dir(cls, output_dir: str) -> "SectionStore":
        import json
        from pathlib import Path

        try:
            tree = json.loads((Path(output_dir) / "section_tree.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        return cls(nodes=tree.get("nodes", {}), root_id=tree.get("root_id"))

    def resolve(self, name: str) -> str | None:
        """node_id của section khớp tên (không phân biệt hoa/thường).

        Ưu tiên title khớp chính xác, sau đó tới title chứa `name`; node sâu hơn thắng.
        Bỏ qua node ROOT (title = tên file PDF) để không match nhầm cả tài liệu.
        """
        kw = (name or "").strip().lower()
        if not kw:
            return None
        exact, partial = [], []
        for nid, n in self.nodes.items():
            title = str(n.get("title", "")).lower()
            if not title or nid == self.root_id:
                continue
            if title == kw:
                exact.append(nid)
            elif kw in title:
                partial.append(nid)
        for pool in (exact, partial):
            if pool:
                return max(pool, key=lambda nid: self.nodes[nid].get("level", 0))
        return None

    def subtree_filter(self, node_id: str):
        """Qdrant filter: chunk thuoc section hoac con chau (ancestor_ids)."""
        from infra.retrieval import build_ancestor_filter  # reuse, lazy qdrant import

        return build_ancestor_filter(node_id)
