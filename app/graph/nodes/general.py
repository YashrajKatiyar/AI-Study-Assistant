import logging
from app.graph.state import StudyAssistantState
from app.llm.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def general_answer_node(state: StudyAssistantState) -> dict:
    llm = get_llm()
    question = state["question"]
    history = state.get("conversation_history", [])
    
    system_prompt = "You are a helpful AI assistant. Answer the user's question directly."
    
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=question))
    
    response = llm.invoke(messages)
    return {"answer": response.content}
