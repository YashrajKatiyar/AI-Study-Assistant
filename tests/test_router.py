from unittest.mock import patch, MagicMock
from app.agents.router_agent import route_query, RouteDecision


@patch("app.agents.router_agent.get_llm")
def test_route_rag_question(mock_get_llm):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouteDecision(
        route="RAG", reason="Question about study material"
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    result = route_query("What is polymorphism?", [])
    assert result.route == "RAG"


@patch("app.agents.router_agent.get_llm")
def test_route_summary_question(mock_get_llm):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouteDecision(
        route="SUMMARY", reason="User wants a summary"
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    result = route_query("Summarize my Java notes", [])
    assert result.route == "SUMMARY"


@patch("app.agents.router_agent.get_llm")
def test_route_general_question(mock_get_llm):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouteDecision(
        route="GENERAL", reason="General knowledge question"
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    result = route_query("What is the capital of France?", [])
    assert result.route == "GENERAL"


@patch("app.agents.router_agent.get_llm")
def test_route_with_conversation_history(mock_get_llm):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouteDecision(
        route="RAG", reason="Follow-up about study material"
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    history = [
        {"role": "user", "content": "What is polymorphism?"},
        {"role": "assistant", "content": "Polymorphism allows..."},
    ]
    result = route_query("Give me a Java example", history)
    assert result.route == "RAG"
    mock_structured.invoke.assert_called_once()


@patch("app.agents.router_agent.get_llm")
def test_route_fallback_on_error(mock_get_llm):
    mock_get_llm.side_effect = Exception("LLM connection failed")

    result = route_query("What is polymorphism?", [])
    assert result.route == "GENERAL"
    assert "failed" in result.reason.lower() or "defaulting" in result.reason.lower()
