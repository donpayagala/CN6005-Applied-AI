"""The tools the agent can call.

Each tool is a normal Python method plus a JSON schema (OpenAI function-calling
format) so the LLM can decide *which* tool to use and *with what arguments*.
Tools only ever return data from the knowledge base, which keeps the LLM's
answers grounded.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

import numpy as np
import pandas as pd

from .retriever import HybridRetriever

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": (
                "Search the movie knowledge base by plot, mood, theme, actor, director or "
                "keywords. Optional filters restrict genre, release years and minimum rating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the user is looking for."},
                    "genre": {"type": "string", "description": "e.g. Comedy, Horror, Science Fiction."},
                    "year_from": {"type": "integer", "description": "Earliest release year."},
                    "year_to": {"type": "integer", "description": "Latest release year."},
                    "min_rating": {"type": "number", "description": "Minimum TMDB rating 0-10."},
                    "k": {"type": "integer", "description": "Number of results (1-10).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_movies",
            "description": "Find movies similar to a movie the user names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the reference movie."},
                    "k": {"type": "integer", "description": "Number of results (1-10).", "default": 5},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": "Get full details (year, genres, director, cast, rating, runtime, plot) of one movie.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Movie title."}},
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_genres",
            "description": "List the genres available in the knowledge base.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _summary(row: pd.Series, overview_chars: int = 220) -> dict[str, Any]:
    """Compact, JSON-friendly view of one movie (keeps prompts short)."""
    overview = row["overview"]
    if len(overview) > overview_chars:
        overview = overview[:overview_chars].rsplit(" ", 1)[0] + "..."
    return {
        "title": row["title"],
        "year": None if pd.isna(row["year"]) else int(row["year"]),
        "genres": list(row["genres"]),
        "director": row["director"],
        "rating": round(float(row["rating"]), 1),
        "overview": overview,
    }


class MovieTools:
    """Callable tools over the knowledge base. Every call is logged in ``self.log``."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.df = retriever.df
        self._titles_lower = self.df["title"].str.lower().tolist()
        self.log: list[dict[str, Any]] = []

    # ----------------------------------------------------------------- helpers
    def _find_index(self, title: str) -> int | None:
        """Exact (case-insensitive) title match first, then closest fuzzy match."""
        title_l = title.strip().lower()
        if title_l in self._titles_lower:
            return self._titles_lower.index(title_l)
        close = difflib.get_close_matches(title_l, self._titles_lower, n=1, cutoff=0.75)
        return self._titles_lower.index(close[0]) if close else None

    def _filter_mask(self, genre=None, year_from=None, year_to=None, min_rating=None) -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)
        if genre:
            g = genre.strip().lower()
            g = {"sci-fi": "science fiction", "scifi": "science fiction", "romcom": "romance"}.get(g, g)
            mask &= self.df["genres"].apply(lambda gs: any(g == x.lower() for x in gs)).to_numpy()
        if year_from:
            mask &= (self.df["year"] >= int(year_from)).to_numpy()
        if year_to:
            mask &= (self.df["year"] <= int(year_to)).to_numpy()
        if min_rating:
            mask &= (self.df["rating"] >= float(min_rating)).to_numpy()
        return mask

    # ------------------------------------------------------------------- tools
    def search_movies(self, query: str, genre: str | None = None, year_from: int | None = None,
                      year_to: int | None = None, min_rating: float | None = None, k: int = 5) -> dict:
        k = max(1, min(int(k or 5), 10))
        mask = self._filter_mask(genre, year_from, year_to, min_rating)
        hits = self.retriever.search(query, k=k, mask=mask)
        return {"results": [_summary(r) for _, r in hits.iterrows()],
                "matched": int(mask.sum())}

    def find_similar_movies(self, title: str, k: int = 5) -> dict:
        k = max(1, min(int(k or 5), 10))
        idx = self._find_index(title)
        if idx is None:
            return {"error": f"'{title}' is not in the knowledge base."}
        scores = self.retriever.similar_to_index(idx).astype(float)
        scores[idx] = -np.inf  # never recommend the movie itself
        order = np.argsort(-scores)[:k]
        return {"reference": self.df.iloc[idx]["title"],
                "results": [_summary(self.df.iloc[i]) for i in order]}

    def get_movie_details(self, title: str) -> dict:
        idx = self._find_index(title)
        if idx is None:
            return {"error": f"'{title}' is not in the knowledge base."}
        row = self.df.iloc[idx]
        details = _summary(row, overview_chars=1000)
        details.update({"cast": list(row["cast"]), "keywords": list(row["keywords"]),
                        "runtime_min": None if pd.isna(row["runtime"]) else int(row["runtime"])})
        return details

    def list_genres(self) -> dict:
        return {"genres": sorted({g for gs in self.df["genres"] for g in gs})}

    # ---------------------------------------------------------------- dispatch
    def execute(self, name: str, arguments: str | dict | None) -> str:
        """Run a tool by name with JSON arguments and return a JSON string."""
        func = {"search_movies": self.search_movies,
                "find_similar_movies": self.find_similar_movies,
                "get_movie_details": self.get_movie_details,
                "list_genres": self.list_genres}.get(name)
        args: Any = arguments
        try:
            args = json.loads(arguments) if isinstance(arguments, str) and arguments else (arguments or {})
            result = func(**args) if func else {"error": f"Unknown tool '{name}'."}
        except Exception as exc:  # bad arguments from the LLM should not crash the agent
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.log.append({"tool": name, "args": args if func else arguments, "result": result})
        return json.dumps(result, ensure_ascii=False)

    def returned_titles(self, since: int = 0) -> list[str]:
        """Titles returned by tool calls (from position ``since`` in the log)."""
        titles: list[str] = []
        for entry in self.log[since:]:
            res = entry["result"]
            titles += [r["title"] for r in res.get("results", [])]
            if "title" in res:
                titles.append(res["title"])
        return list(dict.fromkeys(titles))
