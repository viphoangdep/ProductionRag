"""Context formatting and section depth calculations for retrieved chunks."""
from __future__ import annotations

from typing import Any


def section_label(m: dict[str, Any]) -> str:
    """Human-readable section label from chunk metadata with fallback to source_file."""
    path = m.get("section_path") or ""
    if path and path != "__ROOT__":
        return path
    return m.get("source_file") or m.get("parent_title") or "__ROOT__"


def depth_of(doc: Any, sections: Any = None) -> int:
    """Determine depth level of chunk according to section tree (level property), falling back to section_path."""
    if sections is not None and hasattr(sections, "depth_of"):
        return sections.depth_of(doc)
    m = doc.metadata
    sid = m.get("section_id")
    if sections is not None and sid:
        node = getattr(sections, "nodes", {}).get(sid)
        if node and node.get("level") is not None:
            return int(node["level"])
    path = m.get("section_path") or ""
    if not path or path == "__ROOT__":
        return 0
    return path.count(" > ") + 1


def build_context(docs: list[Any]) -> str:
    """Format a list of documents into an evidence context string for LLM prompting."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        parts.append(
            f"[CTX {i} | {m.get('chunk_id')} | pages={m.get('page_nos')} "
            f"| section={section_label(m)}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)

