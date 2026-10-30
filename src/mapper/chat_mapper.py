from model.res.chat_res import ChatQueryResponse


def to_response(result: dict) -> ChatQueryResponse:
    return ChatQueryResponse(**result)
