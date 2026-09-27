import logging
from langgraph.graph import StateGraph, START, END

from app.graph.state import StudyAssistantState
from app.graph.nodes.router import load_context_node, route_query_node
from app.graph.nodes.rag import (
    rewrite_query_node,
    retrieve_documents_node,
    rerank_documents_node,
    rag_answer_node,
)
from app.graph.nodes.summary import summary_retrieve_node, summary_generate_node
from app.graph.nodes.general import general_answer_node
from app.graph.nodes.final import final_answer_node

logger = logging.getLogger(__name__)


def _route_decision(state: StudyAssistantState) -> str:
    """Conditional edge function: pick the next node based on route."""
    route = state.get("route", "GENERAL")
    if route == "RAG":
        return "rewrite_query"
    elif route == "SUMMARY":
        return "summary_retrieve"
    else:
        return "general_answer"


def build_workflow():
    """Build and compile the LangGraph study-assistant workflow."""
    builder = StateGraph(StudyAssistantState)

    # --- nodes ---
    builder.add_node("load_context", load_context_node)
    builder.add_node("route_query", route_query_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("retrieve_documents", retrieve_documents_node)
    builder.add_node("rerank_documents", rerank_documents_node)
    builder.add_node("rag_answer", rag_answer_node)
    builder.add_node("summary_retrieve", summary_retrieve_node)
    builder.add_node("summary_generate", summary_generate_node)
    builder.add_node("general_answer", general_answer_node)
    builder.add_node("final_answer", final_answer_node)

    # --- edges ---
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "route_query")

    # Conditional routing after the router
    builder.add_conditional_edges(
        "route_query",
        _route_decision,
        {
            "rewrite_query": "rewrite_query",
            "summary_retrieve": "summary_retrieve",
            "general_answer": "general_answer",
        },
    )

    # RAG path
    builder.add_edge("rewrite_query", "retrieve_documents")
    builder.add_edge("retrieve_documents", "rerank_documents")
    builder.add_edge("rerank_documents", "rag_answer")
    builder.add_edge("rag_answer", "final_answer")

    # Summary path
    builder.add_edge("summary_retrieve", "summary_generate")
    builder.add_edge("summary_generate", "final_answer")

    # General path
    builder.add_edge("general_answer", "final_answer")

    # Final → END
    builder.add_edge("final_answer", END)

    graph = builder.compile()
    logger.info("LangGraph workflow compiled successfully")
    return graph
