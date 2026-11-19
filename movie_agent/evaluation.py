"""Evaluation helpers for the movie agent.

Two levels are measured:

1. **Retrieval** - does the search find the right film? (Hit@k, MRR)
   compared across lexical, semantic and hybrid modes.
2. **End-to-end agent** - scenario tests that check tool choice, filter
   constraints, refusal of out-of-scope questions, groundedness and latency.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Description-only queries (no title words) -> the film(s) that should be found.
RETRIEVAL_TESTS: list[tuple[str, set[str]]] = [
    ("thieves who steal secrets by entering people's dreams", {"Inception"}),
    ("toys come alive when humans leave, a cowboy and a space ranger", {"Toy Story"}),
    ("a great white shark terrorises a beach town", {"Jaws"}),
    ("young wizard goes to a school of magic", {"Harry Potter and the Philosopher's Stone"}),
    ("astronauts travel through a wormhole to find a new home for humanity", {"Interstellar"}),
    ("cloned dinosaurs escape in an island theme park", {"Jurassic Park"}),
    ("ageing mafia patriarch hands his crime empire to his reluctant son", {"The Godfather"}),
    ("hacker discovers the world is a computer simulation", {"The Matrix"}),
    ("love story aboard a ship that hits an iceberg", {"Titanic"}),
    ("clownfish father crosses the ocean to find his son", {"Finding Nemo"}),
    ("man with short-term memory loss hunts his wife's killer", {"Memento"}),
    ("insomniac office worker starts an underground fight club", {"Fight Club"}),
    ("banker wrongly imprisoned for murder befriends a fellow inmate", {"The Shawshank Redemption"}),
    ("paraplegic marine on an alien moon joins the native Na'vi", {"Avatar"}),
    ("lonely waste-collecting robot on an abandoned earth", {"WALL·E"}),
    ("emotions inside a young girl's head after her family moves", {"Inside Out"}),
    ("astronaut stranded on Mars grows potatoes to survive", {"The Martian"}),
    ("stockbroker's rise and fall through fraud and excess", {"The Wolf of Wall Street"}),
    ("hobbit sets out to destroy a powerful ring",
     {"The Lord of the Rings: The Fellowship of the Ring", "The Lord of the Rings: The Two Towers",
      "The Lord of the Rings: The Return of the King", "The Hobbit: An Unexpected Journey"}),
    ("Gotham vigilante faces an anarchist clown criminal", {"The Dark Knight"}),
]


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def retrieval_metrics(retriever, tests=RETRIEVAL_TESTS, k: int = 5) -> dict:
    """Hit@1, Hit@k and Mean Reciprocal Rank over the test queries (missing titles skipped)."""
    kb_titles = {_norm(t) for t in retriever.df["title"]}
    hits1, hitsk, rr, used = [], [], [], 0
    for query, relevant in tests:
        relevant_n = {_norm(t) for t in relevant} & kb_titles
        if not relevant_n:
            continue
        used += 1
        ranked = [_norm(t) for t in retriever.search(query, k=k)["title"]]
        rank = next((i + 1 for i, t in enumerate(ranked) if t in relevant_n), None)
        hits1.append(rank == 1)
        hitsk.append(rank is not None)
        rr.append(1 / rank if rank else 0.0)
    return {"queries": used, "hit@1": float(np.mean(hits1)), f"hit@{k}": float(np.mean(hitsk)),
            "MRR": float(np.mean(rr))}


def titles_mentioned(answer: str, kb_titles: list[str]) -> list[str]:
    """Knowledge-base titles that appear in an answer (longer titles first, min 4 chars)."""
    text = answer.lower()
    found = []
    for t in sorted(set(kb_titles), key=len, reverse=True):
        if len(t) < 4:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            found.append(t)
            # Blank out the match so "Toy Story" is not found again inside "Toy Story 2".
            text = re.sub(pattern, lambda m: "#" * len(m.group()), text)
    return found


def groundedness(response, kb_titles: list[str]) -> float | None:
    """Share of KB titles mentioned in the answer that were actually returned by a tool."""
    mentioned = titles_mentioned(response.answer, kb_titles)
    if not mentioned:
        return None
    sources = {_norm(s) for s in response.sources}
    return float(np.mean([_norm(t) in sources for t in mentioned]))


# Each scenario: prompt(s), and checks applied to the final response.
SCENARIOS: list[dict] = [
    {"name": "Plot-based search", "turns": ["I want a mind-bending movie about dreams and heists"],
     "expect_tool": "search_movies"},
    {"name": "Genre + decade filter", "turns": ["Recommend some highly rated science fiction films from the 1980s"],
     "expect_tool": "search_movies", "year_range": (1980, 1989)},
    {"name": "Similar movies", "turns": ["Suggest movies similar to Toy Story"],
     "expect_tool": "find_similar_movies"},
    {"name": "Movie details", "turns": ["Who directed The Dark Knight and how long is it?"],
     "expect_tool": "get_movie_details", "expect_text": ["nolan"]},
    {"name": "Multi-turn memory", "turns": ["Give me a good animated family movie",
                                             "Nice - now something similar to your first suggestion"],
     "expect_tool": "find_similar_movies"},
    {"name": "Out of scope", "turns": ["What is the capital of Australia?"], "expect_no_titles": True},
    {"name": "Unknown film", "turns": ["Tell me about the movie Zorblax Returns 3000"],
     "expect_no_titles": True},
]


def run_scenarios(agent, scenarios=SCENARIOS) -> pd.DataFrame:
    """Run every scenario on a fresh conversation and score it."""
    kb = agent.tools.df
    kb_titles = kb["title"].tolist()
    year_of = dict(zip(kb["title"], kb["year"]))
    rows = []
    for sc in scenarios:
        agent.reset()
        tools_used, latency = [], 0.0
        for turn in sc["turns"]:
            resp = agent.ask(turn)
            tools_used += [c["tool"] for c in resp.tool_calls]
            latency += resp.latency_s
        checks = {}
        if "expect_tool" in sc:
            checks["right_tool"] = sc["expect_tool"] in tools_used
        mentioned = titles_mentioned(resp.answer, kb_titles)
        if "year_range" in sc:
            lo, hi = sc["year_range"]
            checks["filters_respected"] = bool(mentioned) and all(lo <= year_of[t] <= hi for t in mentioned)
        if "expect_text" in sc:
            checks["facts_correct"] = all(s in resp.answer.lower() for s in sc["expect_text"])
        if sc.get("expect_no_titles"):
            checks["declined_correctly"] = not mentioned
        g = groundedness(resp, kb_titles)
        rows.append({"scenario": sc["name"], "passed": all(checks.values()), **checks,
                     "grounded": g, "tools_used": ", ".join(tools_used) or "-",
                     "latency_s": round(latency, 2), "answer": resp.answer})
    agent.reset()
    return pd.DataFrame(rows)
