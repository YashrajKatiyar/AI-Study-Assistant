import logging
from app.graph.state import StudyAssistantState
from app.rag.query_rewriter import rewrite_query
from app.rag.retrieval import retrieve_documents
from app.rag.reranker import rerank_documents
from app.agents.rag_agent import generate_rag_answer

logger = logging.getLogger(__name__)

def rewrite_query_node(state: StudyAssistantState) -> dict:
    question = state["question"]
    history = state.get("conversation_history", [])
    rewritten = rewrite_query(question, history)
    logger.info(f"Original query: '{question}' -> Rewritten: '{rewritten}'")
    return {"rewritten_query": rewritten}

def retrieve_documents_node(state: StudyAssistantState) -> dict:
    query = state.get("rewritten_query", state["question"])
    user_id = state.get("user_id", "default_user")
    results = retrieve_documents(query, user_id)
    logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
    return {"retrieved_documents": results}

def rerank_documents_node(state: StudyAssistantState) -> dict:
    query = state.get("rewritten_query", state["question"])
    docs = state.get("retrieved_documents", [])
    reranked = rerank_documents(query, docs)
    
    # Build context string
    context_parts = [doc.get("content", "") for doc in reranked]
    context_str = "\n\n---\n\n".join(context_parts)
    
    # Build sources list
    sources = []
    for doc in reranked:
        meta = doc.get("metadata", {})
        sources.append({
            "document": meta.get("filename", "Unknown"),
            "page": meta.get("page_number", 1),
            "chunk_id": meta.get("chunk_id", "")
        })
        
    return {
        "retrieved_documents": reranked,
        "context": context_str,
        "sources": sources
    }

def rag_answer_node(state: StudyAssistantState) -> dict:
    question = state["question"]
    context = state.get("context", "")
    history = state.get("conversation_history", [])
    answer = generate_rag_answer(question, context, history)
    return {"answer": answer}
