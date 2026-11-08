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


@router.get("/users/{user_id}/sessions")
def list_user_sessions(user_id: str, service=Depends(deps.get_chat_service)):
    """Lấy danh sách các session đang có của một user."""
    sessions = service.list_user_sessions(user_id=user_id)
    return {"status": "ok", "user_id": user_id, "sessions": sessions}


@router.get("/users/{user_id}/sessions/{session_id}/history")
def get_user_session_history(user_id: str, session_id: str, service=Depends(deps.get_chat_service)):
    """Lấy toàn bộ lịch sử tin nhắn của một session thuộc user."""
    history = service.get_session_history(session_id=session_id, user_id=user_id)
    return {"status": "ok", "user_id": user_id, "session_id": session_id, "history": history}


@router.delete("/users/{user_id}/sessions/{session_id}")
def reset_user_session(user_id: str, session_id: str, service=Depends(deps.get_chat_service)):
    """Xóa một session cụ thể của user."""
    existed = service.reset_session(session_id=session_id, user_id=user_id)
    return {"status": "ok", "user_id": user_id, "session_id": session_id, "existed": existed}


@router.get("/assets/images/{image_id}")
def get_image(image_id: str, assets=Depends(deps.get_asset_store)):
    data, ctype = assets.image_bytes(image_id)
    return Response(content=data, media_type=ctype,
                    headers={"Cache-Control": "private, max-age=300"})
