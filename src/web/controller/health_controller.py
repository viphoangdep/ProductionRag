from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    from web import deps

    return {"status": "ok" if deps._WARMUP["state"] == "ready" else deps._WARMUP["state"],
            "detail": deps._WARMUP["detail"]}
