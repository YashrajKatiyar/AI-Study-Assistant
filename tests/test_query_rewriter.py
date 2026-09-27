from unittest.mock import patch, MagicMock
from app.rag.query_rewriter import rewrite_query


def test_no_rewrite_without_history():
    """If no conversation history, return original question."""
    result = rewrite_query("What is polymorphism?", [])
    assert result == "What is polymorphism?"


def test_no_rewrite_single_message():
    """If only one message in history, return original question."""
    history = [{"role": "user", "content": "Hello"}]
    result = rewrite_query("What is polymorphism?", history)
    assert result == "What is polymorphism?"


@patch("app.rag.query_rewriter.get_llm")
def test_rewrite_with_context(mock_get_llm, sample_conversation_history):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="Give me a Java example of polymorphism"
    )
    mock_get_llm.return_value = mock_llm

    result = rewrite_query("Give me a Java example", sample_conversation_history)
    assert "polymorphism" in result.lower()


@patch("app.rag.query_rewriter.get_llm")
def test_rewrite_fallback_on_error(mock_get_llm, sample_conversation_history):
    mock_get_llm.side_effect = Exception("LLM connection failed")

    result = rewrite_query("Give me a Java example", sample_conversation_history)
    assert result == "Give me a Java example"
