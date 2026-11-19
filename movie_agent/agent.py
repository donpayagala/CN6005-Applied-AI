"""CineGuide agent loop.

    user question
        -> LLM decides: answer directly, or call a tool (search / similar / details / genres)
        -> tool runs against the knowledge base, result goes back to the LLM
        -> repeat (max ``max_steps``) until the LLM writes the final answer
        -> answer + the titles it was based on are returned to the user

Without an LLM configured, a retrieval-only fallback still answers, so the
project can be demonstrated and tested offline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, MovieTools

SYSTEM_PROMPT = """You are CineGuide, a friendly movie recommendation agent.

Rules:
1. Use the tools to look movies up BEFORE recommending. Only recommend or describe
   movies that appear in tool results - never invent titles or facts.
2. Turn the user's request into tool arguments: e.g. "90s sci-fi" -> genre="Science Fiction",
   year_from=1990, year_to=1999. Use find_similar_movies for "movies like X" and
   get_movie_details for questions about one film.
3. Recommend at most 5 movies. For each give the title, year and one short reason
   that is supported by the tool data.
4. If a tool returns nothing useful, say so honestly and suggest how to rephrase.
5. If the request is not about movies, politely explain that you only help with movies.
6. Remember earlier turns of the conversation (e.g. "something shorter", "more like the first one").
"""


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    steps: int = 0


def _assistant_dict(message) -> dict:
    """Convert an SDK message object into a plain dict we can resend."""
    out = {"role": "assistant", "content": message.content or ""}
    if getattr(message, "tool_calls", None):
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in message.tool_calls
        ]
    return out


class MovieAgent:
    def __init__(self, tools: MovieTools, llm=None, max_steps: int = 4,
                 relevance_threshold: float = 0.15):
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
        self.relevance_threshold = relevance_threshold  # used by the offline fallback
        self.reset()

    def reset(self) -> None:
        """Forget the conversation (keeps the knowledge base)."""
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> AgentResponse:
        start, log_start = time.perf_counter(), len(self.tools.log)
        if self.llm is None:
            answer, steps = self._fallback(question), 1
        else:
            answer, steps = self._run_loop(question)
        calls = [{"tool": e["tool"], "args": e["args"]} for e in self.tools.log[log_start:]]
        return AgentResponse(answer=answer, tool_calls=calls,
                             sources=self.tools.returned_titles(since=log_start),
                             latency_s=time.perf_counter() - start, steps=steps)

    # ---------------------------------------------------------------- LLM mode
    def _run_loop(self, question: str) -> tuple[str, int]:
        self.messages.append({"role": "user", "content": question})
        for step in range(1, self.max_steps + 1):
            message = self.llm.chat(self.messages, tools=TOOL_SCHEMAS)
            self.messages.append(_assistant_dict(message))
            if not getattr(message, "tool_calls", None):
                return message.content or "", step
            for call in message.tool_calls:  # act + observe
                result = self.tools.execute(call.function.name, call.function.arguments)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # Step budget used up: ask for a final answer without tools.
        self.messages.append({"role": "user", "content": "Please give your final answer now using the results above."})
        message = self.llm.chat(self.messages)
        self.messages.append({"role": "assistant", "content": message.content or ""})
        return message.content or "", self.max_steps + 1

    # ------------------------------------------------------------ offline mode
    def _fallback(self, question: str) -> str:
        """Retrieval-only answer used when no LLM key is configured."""
        result = json.loads(self.tools.execute("search_movies", {"query": question, "k": 3}))
        hits = result.get("results", [])
        # Raw (unscaled) similarity tells us whether anything is genuinely relevant.
        retriever = self.tools.retriever
        if retriever.embeddings is not None:
            top_score, threshold = float(retriever.semantic_scores(question).max()), 0.30
        else:
            top_score, threshold = float(retriever.lexical_scores(question).max()), self.relevance_threshold
        if not hits or top_score < threshold:
            return ("I could not find matching movies in my knowledge base. "
                    "I can only help with movie recommendations - try describing a plot, genre or actor.")
        lines = [f"- {h['title']} ({h['year']}) - {', '.join(h['genres'][:3])}, rated {h['rating']}/10"
                 for h in hits]
        return "(Offline mode - no LLM configured.) Closest matches:\n" + "\n".join(lines)
