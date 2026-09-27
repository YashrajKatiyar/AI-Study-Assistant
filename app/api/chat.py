import logging
from typing import Any

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.repositories import get_or_create_conversation, add_message
from app.graph.workflow import build_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    question: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    route: str
    sources: list[dict[str, Any]] = []


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Header(default="default_user", alias="X-User-ID"),
    db: Session = Depends(get_db),
):
    """Chat with the AI Study Assistant. Routes through RAG, Summary, or General paths."""
    logger.info(f"Chat request from user {user_id}: {request.question[:80]}")

    # Get or create conversation
    conversation = get_or_create_conversation(db, request.conversation_id, user_id)

    # Save user message
    add_message(db, conversation.id, "user", request.question)

    # Build initial state and run the LangGraph workflow
    initial_state = {
        "user_id": user_id,
        "conversation_id": conversation.id,
        "question": request.question,
    }

    workflow = build_workflow()
    result = workflow.invoke(initial_state)

    # Save assistant response
    answer = result.get("answer", "I'm sorry, I couldn't generate a response.")
    add_message(db, conversation.id, "assistant", answer)

    return ChatResponse(
        conversation_id=conversation.id,
        answer=answer,
        route=result.get("route", "GENERAL"),
        sources=result.get("sources", []),
    )
