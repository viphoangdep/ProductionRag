import json


def safe_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
