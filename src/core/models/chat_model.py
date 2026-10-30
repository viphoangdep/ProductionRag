"""Factory chat model. Đổi provider/model = sửa .env, không sửa code."""


def timeout_kwarg(spec: str, timeout: int) -> dict:
    """Timeout theo từng provider: ollama không có field `timeout` -> đặt ở httpx client."""
    if spec.startswith("ollama:"):
        return {"client_kwargs": {"timeout": timeout}}
    return {"timeout": timeout}


def create_chat_model(spec: str, api_key: str = "", timeout: int | None = None):
    """init_chat_model theo spec 'provider:model'."""
    from langchain.chat_models import init_chat_model  # lazy: nặng, chỉ load lúc warmup

    kw = {"temperature": 0}
    if api_key:
        kw["api_key"] = api_key
    if timeout:
        kw.update(timeout_kwarg(spec, timeout))
    return init_chat_model(spec, **kw)
