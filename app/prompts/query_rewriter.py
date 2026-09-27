QUERY_REWRITER_PROMPT = """
Given a conversation history and a follow-up question from the user, rewrite the follow-up question as a standalone question.
If the question is already self-contained and clear, return it as-is.
Resolve pronouns and references using the conversation context so the rewritten question makes sense without the history.
Do NOT add extra information that is not implied by the conversation history.
Return ONLY the rewritten question, nothing else. Do not wrap it in quotes.
"""
