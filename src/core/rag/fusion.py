"""Pure algorithmic operations for RAG: RRF fusion, parent aggregation, and evidence gating.

All fusion helper formulas and logic have been relocated to `util.fusion` / `util.fusion_util`.
This module re-exports them for backwards compatibility.
"""
from __future__ import annotations

from util.fusion_util import (
    aggregate_parents,
    group_key,
    group_title,
    is_evidence_sufficient,
    rrf_fuse,
    top_parents_expand,
)

__all__ = [
    "rrf_fuse",
    "group_key",
    "group_title",
    "aggregate_parents",
    "top_parents_expand",
    "is_evidence_sufficient",
]
