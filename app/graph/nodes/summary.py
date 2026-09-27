import logging
from app.graph.state import StudyAssistantState
from app.rag.retrieval import retrieve_document_chunks, retrieve_documents
from app.agents.summary_agent import generate_summary

logger = logging.getLogger(__name__)

def summary_retrieve_node(state: StudyAssistantState) -> dict:
    user_id = state.get("user_id", "default_user")
    document_id = state.get("document_id")
    
    if document_id:
        results = retrieve_document_chunks(document_id, user_id)
        logger.info(f"Retrieved chunks for document_id {document_id}")
    else:
        question = state["question"]
        results = retrieve_documents(question, user_id)
        logger.info(f"Retrieved topic-based documents for summary")
        
    return {"retrieved_documents": results}

def summary_generate_node(state: StudyAssistantState) -> dict:
    docs = state.get("retrieved_documents", [])
    question = state["question"]
    answer = generate_summary(docs, question)
    return {"answer": answer}
