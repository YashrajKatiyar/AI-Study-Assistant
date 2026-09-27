import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings
from app.llm.llm import get_llm

logger = logging.getLogger(__name__)

RERANKER_SYSTEM_PROMPT = """You are a document relevance scorer. Given a query and a list of document chunks, score each chunk's relevance to the query from 0 to 10. Return ONLY a JSON array of objects with 'index' (0-based) and 'score' fields. Example: [{"index": 0, "score": 8.5}, {"index": 1, "score": 3.0}]"""

def rerank_documents(query: str, documents: List[Dict[str, Any]], final_k: Optional[int] = None) -> List[Dict[str, Any]]:
    k = final_k if final_k is not None else get_settings().RAG_FINAL_K
    
    if len(documents) <= k:
        return documents
        
    try:
        llm = get_llm(temperature=0.0)
        
        chunks_text = ""
        for idx, doc in enumerate(documents):
            chunks_text += f"--- Document {idx} ---\n{doc['content']}\n\n"
            
        messages = [
            SystemMessage(content=RERANKER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nDocuments:\n{chunks_text}")
        ]
        
        response = llm.invoke(messages)
        content = str(response.content).strip()
        
        # Clean up JSON formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        scores = json.loads(content.strip())
        
        scored_docs = []
        for item in scores:
            idx = item.get("index")
            score = item.get("score", 0.0)
            if idx is not None and 0 <= idx < len(documents):
                scored_docs.append((score, documents[idx]))
                
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs][:k]
        
    except Exception as e:
        logger.warning(f"Reranking failed: {str(e)}. Using top {k} original documents.")
        return documents[:k]
