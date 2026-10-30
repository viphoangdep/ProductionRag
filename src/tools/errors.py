"""Tool error handling middleware (langchain docs pattern).

Convert tool exceptions into ToolMessages the model can handle instead of
crashing the agent loop.
"""
from collections.abc import Callable

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest


@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Convert tool exceptions into ToolMessages the model can handle."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Document search encountered an error: {e}. Please inform the user that document search is currently unavailable.",
            tool_call_id=request.tool_call["id"],
        )
