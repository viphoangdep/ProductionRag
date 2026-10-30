from fastapi import APIRouter, Depends, Response

from model.req.chat_req import ChatQueryRequest
from model.res.chat_res import ChatQueryResponse
from web import deps

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/query")
def chat_query(body: ChatQueryRequest, service=Depends(deps.get_chat_service)) -> ChatQueryResponse:
    return service.query(body)


@router.delete("/sessions/{session_id}")
def reset_session(session_id: str, service=Depends(deps.get_chat_service)):
    """Reset lịch sử cuộc trò chuyện của một session_id."""
    existed = service.reset_session(session_id)
    return {"status": "ok", "session_id": session_id, "existed": existed}


@router.get("/assets/images/{image_id}")
def get_image(image_id: str, assets=Depends(deps.get_asset_store)):
    data, ctype = assets.image_bytes(image_id)
    return Response(content=data, media_type=ctype,
                    headers={"Cache-Control": "private, max-age=300"})
