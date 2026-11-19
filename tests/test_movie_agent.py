"""Offline unit tests for the movie agent (no API key, no downloads).

Run with:  pytest -q
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from movie_agent import HybridRetriever, MovieAgent, MovieTools
from movie_agent.data import prepare_movies
from movie_agent.evaluation import groundedness, retrieval_metrics, titles_mentioned


def _g(*names):
    return json.dumps([{"id": i, "name": n} for i, n in enumerate(names)])


RAW = [
    # title, year, genres, keywords, overview, rating, votes, runtime, director, cast
    ("Inception", "2010-07-14", ("Action", "Science Fiction"), ("dream", "heist", "subconscious"),
     "Cobb, a skilled thief who steals secrets from deep within the subconscious during the dream state, "
     "is offered a chance to have his criminal history erased.", 8.1, 13752, 148, "Christopher Nolan",
     ("Leonardo DiCaprio", "Joseph Gordon-Levitt")),
    ("Toy Story", "1995-10-30", ("Animation", "Comedy", "Family"), ("toy", "cowboy", "friendship"),
     "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear "
     "onto the scene.", 7.7, 5269, 81, "John Lasseter", ("Tom Hanks", "Tim Allen")),
    ("Toy Story 2", "1999-10-30", ("Animation", "Comedy", "Family"), ("toy", "cowboy", "collector"),
     "Andy heads off to Cowboy Camp, leaving his toys to their own devices, and Woody is kidnapped "
     "by a toy collector.", 7.3, 3767, 92, "John Lasseter", ("Tom Hanks", "Tim Allen")),
    ("Jaws", "1975-06-18", ("Horror", "Thriller", "Adventure"), ("shark", "beach", "island"),
     "An insatiable great white shark terrorizes the townspeople of Amity Island.", 7.5, 3170, 124,
     "Steven Spielberg", ("Roy Scheider", "Robert Shaw")),
    ("The Terminator", "1984-10-26", ("Action", "Thriller", "Science Fiction"), ("cyborg", "future", "time travel"),
     "In the post-apocalyptic future, a cyborg assassin is sent back in time to kill Sarah Connor.",
     7.3, 4128, 108, "James Cameron", ("Arnold Schwarzenegger", "Linda Hamilton")),
    ("Aliens", "1986-07-18", ("Horror", "Action", "Thriller", "Science Fiction"), ("alien", "marine", "space"),
     "Ripley returns to the planet LV-426 with a unit of space marines to fight the alien creatures.",
     7.7, 3220, 137, "James Cameron", ("Sigourney Weaver", "Michael Biehn")),
    ("The Notebook", "2004-06-25", ("Romance", "Drama"), ("love letter", "memory"),
     "An elderly man reads to a woman with dementia the story of two young lovers in the 1940s.",
     7.7, 3067, 123, "Nick Cassavetes", ("Ryan Gosling", "Rachel McAdams")),
    ("Obscure Film", "2001-01-01", ("Drama",), (), "Too few votes to be kept.", 5.0, 3, 90, "Nobody", ()),
]


@pytest.fixture(scope="module")
def kb():
    movies = pd.DataFrame([{
        "id": i, "title": t, "release_date": d, "genres": _g(*g), "keywords": _g(*k), "overview": o,
        "vote_average": r, "vote_count": v, "runtime": rt} for i, (t, d, g, k, o, r, v, rt, _, _) in enumerate(RAW)])
    credits = pd.DataFrame([{
        "movie_id": i, "title": t,
        "cast": json.dumps([{"name": c, "order": j} for j, c in enumerate(cast)]),
        "crew": json.dumps([{"job": "Director", "name": dr}])} for i, (t, *_, dr, cast) in enumerate(RAW)])
    return prepare_movies(movies, credits, top_n=100, min_votes=100)


@pytest.fixture()
def tools(kb):
    return MovieTools(HybridRetriever(kb, mode="lexical"))


def test_prepare_filters_and_parses(kb):
    assert "Obscure Film" not in set(kb["title"])  # removed by the vote filter
    inception = kb[kb["title"] == "Inception"].iloc[0]
    assert inception["year"] == 2010
    assert inception["director"] == "Christopher Nolan"
    assert "Science Fiction" in inception["genres"]
    assert "Leonardo DiCaprio" in inception["doc"]


def test_search_ranks_relevant_movie_first(tools):
    result = tools.search_movies("great white shark attacks a beach island")
    assert result["results"][0]["title"] == "Jaws"


def test_search_filters(tools):
    result = tools.search_movies("cyborg alien future", genre="sci-fi", year_from=1980, year_to=1989)
    titles = {r["title"] for r in result["results"]}
    assert titles == {"The Terminator", "Aliens"}
    assert tools.search_movies("love", min_rating=7.6, genre="Romance")["results"][0]["title"] == "The Notebook"


def test_similar_excludes_itself_and_fuzzy_title(tools):
    result = tools.find_similar_movies("toy story", k=2)
    assert result["reference"] == "Toy Story"
    assert result["results"][0]["title"] == "Toy Story 2"
    assert all(r["title"] != "Toy Story" for r in result["results"])
    assert tools.get_movie_details("Incepton")["title"] == "Inception"  # typo tolerated


def test_unknown_movie_and_bad_tool_do_not_crash(tools):
    assert "error" in tools.get_movie_details("Zorblax Returns 3000")
    assert "error" in json.loads(tools.execute("delete_database", "{}"))
    assert "error" in json.loads(tools.execute("search_movies", "{not json"))


def test_offline_agent_answers_and_declines(tools):
    agent = MovieAgent(tools, llm=None)
    resp = agent.ask("shark attacks at the beach")
    assert "Jaws" in resp.answer and "Jaws" in resp.sources
    assert "could not find" in agent.ask("capital of Australia").answer


class FakeLLM:
    """Scripted LLM: first asks for a tool, then answers using the tool output."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            call = SimpleNamespace(id="call_1", function=SimpleNamespace(
                name="find_similar_movies", arguments=json.dumps({"title": "Toy Story", "k": 1})))
            return SimpleNamespace(content=None, tool_calls=[call])
        observation = json.loads(messages[-1]["content"])
        title = observation["results"][0]["title"]
        return SimpleNamespace(content=f"You might enjoy {title}.", tool_calls=None)


def test_agent_tool_loop_with_fake_llm(tools):
    agent = MovieAgent(tools, llm=FakeLLM())
    resp = agent.ask("Something like Toy Story?")
    assert resp.answer == "You might enjoy Toy Story 2."
    assert resp.tool_calls[0]["tool"] == "find_similar_movies"
    assert resp.steps == 2
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert groundedness(resp, tools.df["title"].tolist()) == 1.0


class BrokenLLM:
    def chat(self, messages, tools=None):
        raise RuntimeError("rate limited")


def test_llm_failure_is_reported_not_raised(tools):
    agent = MovieAgent(tools, llm=BrokenLLM())
    resp = agent.ask("Something like Toy Story?")
    assert "failed to respond" in resp.answer and "rate limited" in resp.answer
    assert [m["role"] for m in agent.messages] == ["system"]  # broken turn removed
    agent.llm = FakeLLM()
    assert agent.ask("Something like Toy Story?").answer == "You might enjoy Toy Story 2."


def test_titles_mentioned_and_retrieval_metrics(tools):
    titles = tools.df["title"].tolist()
    assert titles_mentioned("Try Toy Story 2 or Jaws!", titles) == ["Toy Story 2", "Jaws"]
    m = retrieval_metrics(tools.retriever, tests=[("shark beach", {"Jaws"}), ("dream heist", {"Inception"})])
    assert m["queries"] == 2 and m["hit@1"] == 1.0 and m["MRR"] == 1.0
