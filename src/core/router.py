"""Dynamic router: rules rẻ, không tốn hop LLM yếu.

SIMPLE -> weak model + tools=[] + prompt ngắn;
else -> RAG (strong + [search_documents] + prompt RAG).
Sau này nhiều task thì đổi `else` thành TaskType.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import SystemMessage

from config.prompts import SIMPLE_PROMPT, SYSTEM_PROMPT
from tools.search_tool import search_documents
from util.string_util import normalize_ws

log = logging.getLogger("ai-service")

_SIMPLE = re.compile(
    r"^(xin ch[aà]o|ch[aà]o|hello|hi|hey|yo|b[aạ]n l[aà] ai|who are you|"
    r"t[eê]n (c[uủ]a )?(t[oô]i|b[aạ]n)|what is (your|my) name|my name is|"
    r"t[oô]i l[aà]|m[iì]nh l[aà]|i am|"
    r"c[aả]m [oơ]n|thank|thanks|t[aạ]m bi[eệ]t|bye)\b",
    re.IGNORECASE,
)


def _last_text(messages: list) -> str:
    if not messages:
        return ""
    c = getattr(messages[-1], "content", messages[-1])
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # content blocks
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


def is_simple(query_norm: str) -> bool:
    q = query_norm.strip()
    return len(q) < 50 and bool(_SIMPLE.search(q))


def pick_prompt(query_norm: str) -> str:
    return SIMPLE_PROMPT if is_simple(query_norm) else SYSTEM_PROMPT


@wrap_model_call
def dynamic_router(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    qn = normalize_ws(_last_text(request.messages))
    ctx = request.runtime.context or {}
    if is_simple(qn):
        log.info("route=SIMPLE model=%s q=%.100s", ctx.get("weak_name", "?"), qn)
        return handler(request.override(model=ctx["weak_model"], tools=[],
                                        system_message=SystemMessage(content=SIMPLE_PROMPT)))
    # ponytail: else -> RAG; tách TaskType enum khi có thêm SQL_QUERY
    log.info("route=RAG model=%s q=%.100s", ctx.get("strong_name", "?"), qn)
    return handler(request.override(model=ctx["strong_model"], tools=[search_documents],
                                    system_message=SystemMessage(content=SYSTEM_PROMPT)))
