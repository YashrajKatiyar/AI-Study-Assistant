import logging
from app.graph.state import StudyAssistantState
from app.agents.final_agent import generate_final_answer

logger = logging.getLogger(__name__)


def final_answer_node(state: StudyAssistantState) -> dict:
    answer = state.get("answer", "")
    route = state.get("route", "GENERAL")
    question = state["question"]
    final = generate_final_answer(answer, route, question)
    logger.info(f"Final answer generated for route: {route}")
    return {"answer": final}
