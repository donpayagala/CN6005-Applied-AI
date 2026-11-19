"""Command-line chat:  python -m movie_agent   (type 'reset' or 'quit')."""

from . import HybridRetriever, MovieAgent, MovieTools, load_knowledge_base
from .llm import make_client


def main() -> None:
    print("Loading knowledge base...")
    retriever = HybridRetriever(load_knowledge_base())
    llm = make_client()
    agent = MovieAgent(MovieTools(retriever), llm=llm)
    mode = f"{llm.provider} / {llm.model}" if llm else "offline (no API key found)"
    print(f"CineGuide ready - {len(retriever.df)} movies, LLM: {mode}\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "reset":
            agent.reset()
            print("(conversation cleared)\n")
            continue
        if question:
            resp = agent.ask(question)
            tools = ", ".join(c["tool"] for c in resp.tool_calls) or "none"
            print(f"\nCineGuide: {resp.answer}\n   [tools: {tools} | {resp.latency_s:.1f}s]\n")


if __name__ == "__main__":
    main()
