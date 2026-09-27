import logging
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from app.llm.llm import get_llm
from app.prompts.query_rewriter import QUERY_REWRITER_PROMPT

logger = logging.getLogger(__name__)

def rewrite_query(question: str, conversation_history: List[Dict[str, str]]) -> str:
    if not conversation_history or len(conversation_history) < 2:
        return question
        
    try:
        llm = get_llm(temperature=0.1)
        
        formatted_history = ""
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_history += f"{role}: {content}\n"
            
        messages = [
            SystemMessage(content=QUERY_REWRITER_PROMPT),
            HumanMessage(content=f"Conversation History:\n{formatted_history}\n\nCurrent Question: {question}")
        ]
        
        response = llm.invoke(messages)
        return str(response.content).strip()
    except Exception as e:
        logger.warning(f"Failed to rewrite query: {str(e)}. Using original question.")
        return question
