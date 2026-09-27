import logging

from langchain_core.messages import SystemMessage, HumanMessage

from app.llm.llm import get_llm
from app.prompts.final import FINAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def generate_final_answer(answer: str, route: str, question: str) -> str:
    try:
        llm = get_llm(temperature=0.1)
        
        messages = [
            SystemMessage(content=FINAL_SYSTEM_PROMPT),
            HumanMessage(content=f"Original question: {question}\nRoute: {route}\nDraft answer:\n{answer}\n\nPlease provide a polished final response.")
        ]
        
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as e:
        logger.warning(f"Final agent generation failed: {str(e)}. Returning original answer.")
        return answer
