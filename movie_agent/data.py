"""Knowledge-base loading for the movie agent.

The knowledge source is the public Kaggle "TMDB 5000 Movie Dataset"
(https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). We keep the most
voted films so the knowledge base stays small, well-known and reliable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "tmdb/tmdb-movie-metadata"
MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_5000_credits.csv"


def _names(json_text: str, limit: int | None = None) -> list[str]:
    """Turn a TMDB JSON list such as '[{"id": 28, "name": "Action"}]' into ['Action']."""
    try:
        items = json.loads(json_text) if isinstance(json_text, str) else []
    except json.JSONDecodeError:
        return []
    names = [item["name"] for item in items if "name" in item]
    return names[:limit] if limit else names


def _director(crew_json: str) -> str:
    """Return the first crew member whose job is 'Director' (empty string if none)."""
    try:
        crew = json.loads(crew_json) if isinstance(crew_json, str) else []
    except json.JSONDecodeError:
        return ""
    return next((member["name"] for member in crew if member.get("job") == "Director"), "")


def build_document(row: pd.Series) -> str:
    """Build the text that the retriever indexes for one movie.

    Putting title, genres, people, keywords and the plot into one string lets a
    single search match questions about any of them.
    """
    year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
    return (
        f"{row['title']}{year}. "
        f"Genres: {', '.join(row['genres'])}. "
        f"Director: {row['director']}. "
        f"Starring: {', '.join(row['cast'])}. "
        f"Keywords: {', '.join(row['keywords'])}. "
        f"Plot: {row['overview']}"
    )


def prepare_movies(movies: pd.DataFrame, credits: pd.DataFrame | None = None,
                   top_n: int = 1000, min_votes: int = 100) -> pd.DataFrame:
    """Clean the raw TMDB tables into the knowledge-base table used by the agent."""
    df = movies.copy()
    if credits is not None:
        credits = credits.rename(columns={"movie_id": "id"})[["id", "cast", "crew"]]
        df = df.merge(credits, on="id", how="left")

    df["genres"] = df["genres"].apply(_names)
    df["keywords"] = df["keywords"].apply(lambda s: _names(s, limit=12))
    df["cast"] = df["cast"].apply(lambda s: _names(s, limit=4)) if "cast" in df else [[]] * len(df)
    df["director"] = df["crew"].apply(_director) if "crew" in df else ""
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["overview"] = df["overview"].fillna("")

    # Keep released, reasonably well-known films with a plot description.
    df = df[(df["vote_count"] >= min_votes) & (df["overview"].str.len() > 20)]
    df = df.sort_values("vote_count", ascending=False).head(top_n)

    df = df.rename(columns={"vote_average": "rating"})
    df["doc"] = df.apply(build_document, axis=1)
    columns = ["id", "title", "year", "genres", "director", "cast", "keywords",
               "rating", "vote_count", "runtime", "overview", "doc"]
    return df[columns].reset_index(drop=True)


def load_knowledge_base(data_dir: str | os.PathLike | None = None, top_n: int = 1000) -> pd.DataFrame:
    """Load the TMDB CSVs (downloading them with kagglehub if no folder is given)."""
    data_dir = data_dir or os.environ.get("TMDB_DATA_DIR")
    if data_dir is None:
        import kagglehub  # imported lazily so the tests do not need it

        data_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    data_dir = Path(data_dir)
    movies = pd.read_csv(data_dir / MOVIES_FILE)
    credits_path = data_dir / CREDITS_FILE
    credits = pd.read_csv(credits_path) if credits_path.exists() else None
    return prepare_movies(movies, credits, top_n=top_n)
