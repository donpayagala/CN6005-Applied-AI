"""CineGuide - a tool-using movie recommendation agent (CN6005 Task 3).

The package is split into small, testable modules:

* ``data``       - loads and cleans the TMDB 5000 knowledge base
* ``retriever``  - hybrid lexical (TF-IDF) + semantic (Sentence-BERT) search
* ``tools``      - the functions the agent is allowed to call
* ``llm``        - a thin, provider-agnostic LLM client (Groq, Gemini, OpenAI, Claude, Ollama)
* ``agent``      - the plan -> act (tool call) -> observe -> answer loop
* ``evaluation`` - retrieval metrics and end-to-end scenario tests
"""

from .agent import AgentResponse, MovieAgent
from .data import load_knowledge_base
from .retriever import HybridRetriever
from .tools import MovieTools

__all__ = [
    "AgentResponse",
    "HybridRetriever",
    "MovieAgent",
    "MovieTools",
    "load_knowledge_base",
]
