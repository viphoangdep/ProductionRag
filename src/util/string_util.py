import re


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_paraphrases(lines: list[str], query_norm: str, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        line = line.strip().lstrip("0123456789.-) ").strip()
        if not line or line.lower() == query_norm.lower() or line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= limit:
            break
    return out
