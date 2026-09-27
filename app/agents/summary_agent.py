import logging
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.exceptions import LLMError
from app.llm.llm import get_llm
from app.prompts.summary import SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def generate_summary(documents: List[Dict[str, Any]], question: str) -> str:
    try:
        llm = get_llm(temperature=0.1)
        
        context = "\n\n".join([doc.get('content', '') for doc in documents])
        
        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=f"Study material to summarize:\n{context}\n\nRequest: {question}")
        ]
        
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as e:
        logger.error(f"Summary agent generation failed: {str(e)}")
        raise LLMError(f"Failed to generate summary: {str(e)}") from e
