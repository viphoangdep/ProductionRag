"""In-app lightweight security & policy guards for RAG.

Includes:
- English prompt injection patterns and detection.
- Tool budget counter to prevent runaway LLM agent loops.
"""
from __future__ import annotations

import re
import logging
from core.exception.errors import AppError, ErrorCode

log = logging.getLogger("ai-service")

# Common English Prompt Injection & Jailbreak patterns
INJECTION_PATTERNS = [
    # Override system instruction
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|context)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|context)\b",
    r"(?i)\bforget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|context)\b",
    # System roleplay & jailbreak attempts
    r"(?i)\byou\s+are\s+now\s+(DAN|unrestricted|jailbroken|an\s+AI\s+without\s+rules|in\s+developer\s+mode)\b",
    r"(?i)\bswitch\s+to\s+(developer|god|unrestricted|unfiltered)\s+mode\b",
    r"(?i)\bpretend\s+you\s+(have\s+no\s+rules|can\s+do\s+anything|are\s+an\s+evil\s+AI)\b",
    # System prompt extraction
    r"(?i)\b(show|reveal|display|output|print|repeat)\s+(your\s+)?(system\s+prompt|initial\s+prompt|core\s+instructions|system\s+instructions)\b",
    r"(?i)\bwhat\s+(is|are)\s+your\s+((original|system|secret)\s*)*(instructions|prompt|rules)\b",
    # Special system / role markers
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
    r"(?i)\[INST\]",
    r"(?i)\[/INST\]",
    r"(?i)###\s*(System|Instruction|Assistant):",
]

_COMPILED_PATTERNS = [re.compile(p) for p in INJECTION_PATTERNS]


def check_prompt_injection(text: str) -> None:
    """Check text against English prompt injection patterns.
    
    Raises:
        AppError with PROMPT_INJECTION_DETECTED if a match is found.
    """
    if not text:
        return
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_str = match.group(0)
            log.warning("Prompt injection detected pattern='%s' text='%.80s'", matched_str, text)
            raise AppError(
                ErrorCode.PROMPT_INJECTION_DETECTED,
                f"Prompt injection detected in input: matched pattern '{matched_str}'",
            )


class ToolBudget:
    """Lightweight tool invocation counter and budget tracker."""

    def __init__(self, max_calls: int = 3):
        self.max_calls = max_calls
        self.current_calls = 0

    def consume(self, tool_name: str = "tool") -> None:
        """Increment count and check against limit."""
        self.current_calls += 1
        if self.current_calls > self.max_calls:
            log.warning(
                "Tool budget exceeded tool=%s calls=%d max=%d",
                tool_name, self.current_calls, self.max_calls
            )
            raise AppError(
                ErrorCode.TOOL_BUDGET_EXCEEDED,
                f"Tool invocation limit exceeded (max {self.max_calls} calls)",
            )
