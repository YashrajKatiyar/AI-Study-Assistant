from typing import TypedDict, Any

class StudyAssistantState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    question: str
    rewritten_query: str
    route: str  # "RAG" | "SUMMARY" | "GENERAL"
    retrieved_documents: list[Any]
    context: str
    conversation_history: list[dict[str, str]]
    answer: str
    sources: list[dict[str, Any]]
    document_id: str  # For summary requests
