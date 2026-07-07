"""
routers/chat.py
----------------
Anthropic-powered petrophysics assistant endpoint (section 5 of the brief).

Streams Server-Sent Events (SSE) so the frontend chat panel can render
tokens as they arrive, plus a running log of tool calls the agent makes
(for transparency into what data grounded its answer).
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest
from app.services.anthropic_agent import stream_chat_response

router = APIRouter(tags=["chat"])


def _format_sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Accepts {message, well_id?, conversation_history} and streams back
    the agent's response as Server-Sent Events.

    Event types sent to the client:
      - {"type": "text_delta", "text": "..."}       incremental assistant text
      - {"type": "tool_call", "name", "input", "output"}  a tool the agent invoked
      - {"type": "done"}                              stream complete
      - {"type": "error", "message": "..."}           something went wrong
    """
    history = [m.model_dump() for m in request.conversation_history]

    async def event_generator():
        async for event in stream_chat_response(
            request.message, request.well_id, history
        ):
            yield _format_sse(event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
