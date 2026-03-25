import ollama
from .config import OLLAMA_MODEL

def get_standalone_query(query: str, history: list):
    if not history:
        return query

    # Format history for the LLM
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    prompt = f"""Given the following conversation and a follow-up question, re-phrase the follow-up question to be a standalone question.
    Chat History:
    {history_str}
    Follow-up Question: {query}
    Standalone Question:"""

    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
        return response['response'].strip()
    except:
        return query

def generate_multi_queries(query: str, count: int = 2):
    prompt = f"""Generate {count} different versions of the following user query to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user query, your goal is to help the user overcome some of the limitations of distance-based similarity search.
    Provide these alternative queries separated by newlines.
    Original query: {query}"""
    
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
        queries = [q.strip() for q in response['response'].split('\n') if q.strip()]
        return [query] + queries[:count]
    except:
        return [query]