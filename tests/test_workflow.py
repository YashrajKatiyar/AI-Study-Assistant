from unittest.mock import patch, MagicMock
from app.agents.router_agent import RouteDecision
from app.graph.workflow import build_workflow


def test_workflow_compiles():
    """The LangGraph workflow should compile without errors."""
    graph = build_workflow()
    assert graph is not None


@patch("app.graph.nodes.final.generate_final_answer")
@patch("app.graph.nodes.rag.generate_rag_answer")
@patch("app.graph.nodes.rag.rerank_documents")
@patch("app.graph.nodes.rag.retrieve_documents")
@patch("app.graph.nodes.rag.rewrite_query")
@patch("app.graph.nodes.router.route_query")
@patch("app.graph.nodes.router.get_recent_messages")
@patch("app.graph.nodes.router.SessionLocal")
def test_rag_path(
    mock_session, mock_messages, mock_route,
    mock_rewrite, mock_retrieve, mock_rerank,
    mock_rag_answer, mock_final,
):
    mock_db = MagicMock()
    mock_session.return_value = mock_db
    mock_messages.return_value = []
    mock_route.return_value = RouteDecision(route="RAG", reason="test")
    mock_rewrite.return_value = "rewritten query"
    mock_retrieve.return_value = [
        {"content": "test content", "metadata": {"filename": "test.pdf", "page_number": 1, "chunk_id": "c1"}}
    ]
    mock_rerank.return_value = [
        {"content": "test content", "metadata": {"filename": "test.pdf", "page_number": 1, "chunk_id": "c1"}}
    ]
    mock_rag_answer.return_value = "RAG answer"
    mock_final.return_value = "Final RAG answer"

    workflow = build_workflow()
    result = workflow.invoke({
        "user_id": "test",
        "conversation_id": "conv1",
        "question": "What is polymorphism?",
    })
    assert result["route"] == "RAG"
    assert result["answer"] == "Final RAG answer"


@patch("app.graph.nodes.final.generate_final_answer")
@patch("app.graph.nodes.summary.generate_summary")
@patch("app.graph.nodes.summary.retrieve_document_chunks")
@patch("app.graph.nodes.router.route_query")
@patch("app.graph.nodes.router.get_recent_messages")
@patch("app.graph.nodes.router.SessionLocal")
def test_summary_path(
    mock_session, mock_messages, mock_route,
    mock_retrieve_chunks, mock_summary, mock_final,
):
    mock_db = MagicMock()
    mock_session.return_value = mock_db
    mock_messages.return_value = []
    mock_route.return_value = RouteDecision(route="SUMMARY", reason="test")
    mock_retrieve_chunks.return_value = [
        {"content": "chapter content", "metadata": {"filename": "notes.pdf", "page_number": 1, "chunk_id": "c1"}}
    ]
    mock_summary.return_value = "Summary of the document"
    mock_final.return_value = "Final summary"

    workflow = build_workflow()
    result = workflow.invoke({
        "user_id": "test",
        "conversation_id": "conv1",
        "question": "Summarize my notes",
        "document_id": "doc1",
    })
    assert result["route"] == "SUMMARY"
    assert result["answer"] == "Final summary"


@patch("app.graph.nodes.final.generate_final_answer")
@patch("app.graph.nodes.general.get_llm")
@patch("app.graph.nodes.router.route_query")
@patch("app.graph.nodes.router.get_recent_messages")
@patch("app.graph.nodes.router.SessionLocal")
def test_general_path(
    mock_session, mock_messages, mock_route,
    mock_get_llm, mock_final,
):
    mock_db = MagicMock()
    mock_session.return_value = mock_db
    mock_messages.return_value = []
    mock_route.return_value = RouteDecision(route="GENERAL", reason="test")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Paris is the capital of France")
    mock_get_llm.return_value = mock_llm
    mock_final.return_value = "Paris is the capital of France."

    workflow = build_workflow()
    result = workflow.invoke({
        "user_id": "test",
        "conversation_id": "conv1",
        "question": "What is the capital of France?",
    })
    assert result["route"] == "GENERAL"
    assert result["answer"] == "Paris is the capital of France."
