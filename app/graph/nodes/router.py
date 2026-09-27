import logging
from app.graph.state import StudyAssistantState
from app.core.config import get_settings
from app.database.database import SessionLocal
from app.database.repositories import get_recent_messages
from app.agents.router_agent import route_query

logger = logging.getLogger(__name__)

def load_context_node(state: StudyAssistantState) -> dict:
    conversation_id = state.get("conversation_id")
    if conversation_id:
        db = SessionLocal()
        try:
            messages = get_recent_messages(db, conversation_id, get_settings().MAX_CONVERSATION_HISTORY)
            history = [{"role": m.role, "content": m.content} for m in messages]
            return {"conversation_history": history}
        finally:
            db.close()
    return {"conversation_history": []}

def route_query_node(state: StudyAssistantState) -> dict:
    question = state["question"]
    history = state.get("conversation_history", [])
    decision = route_query(question, history)
    logger.info(f"Routing decision for query: {decision.route}")
    return {"route": decision.route}
