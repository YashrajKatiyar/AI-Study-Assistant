import logging
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.exceptions import LLMError
from app.llm.llm import get_llm
from app.prompts.rag import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def generate_rag_answer(question: str, context: str, conversation_history: List[Dict[str, str]]) -> str:
    try:
        llm = get_llm(temperature=0.1)
        
        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
        
        for msg in conversation_history[-5:]: # reasonable limit
            role = msg.get("role", "unknown").lower()
            content = msg.get("content", "")
            if role in ("user", "human"):
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=content))
                
        messages.append(HumanMessage(content=f"Context from study material:\n{context}\n\nQuestion: {question}"))
        
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as e:
        logger.error(f"RAG agent generation failed: {str(e)}")
        raise LLMError(f"Failed to generate RAG answer: {str(e)}") from e
