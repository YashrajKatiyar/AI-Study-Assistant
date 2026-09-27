import logging
from typing import Literal, List, Dict

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from app.llm.llm import get_llm
from app.prompts.router import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RouteDecision(BaseModel):
    route: Literal["RAG", "SUMMARY", "GENERAL"] = Field(description="The type of query")
    reason: str = Field(description="Brief explanation for the routing decision")

def route_query(question: str, conversation_history: List[Dict[str, str]]) -> RouteDecision:
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(RouteDecision)
        
        formatted_history = ""
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_history += f"{role}: {content}\n"
            
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Conversation History:\n{formatted_history}\n\nQuestion: {question}")
        ]
        
        result = structured_llm.invoke(messages)
        logger.info(f"Routed query to {result.route}: {result.reason}")
        return result
    except Exception as e:
        logger.error(f"Routing failed: {str(e)}")
        return RouteDecision(route="GENERAL", reason="Routing failed, defaulting to general")
