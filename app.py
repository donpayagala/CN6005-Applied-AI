"""Streamlit chat UI for the CineGuide movie agent.

Run:  streamlit run app.py
"""

import os

import streamlit as st

from movie_agent import HybridRetriever, MovieAgent, MovieTools, load_knowledge_base
from movie_agent.llm import PROVIDERS, detect_provider, make_client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="CineGuide", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading the movie knowledge base and embedding model...")
def get_retriever() -> HybridRetriever:
    return HybridRetriever(load_knowledge_base(top_n=1000), mode="hybrid")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    available = [p for p, cfg in PROVIDERS.items() if not cfg["key_env"] or os.environ.get(cfg["key_env"])]
    default = detect_provider()
    options = available + ["offline"]
    provider = st.selectbox("LLM provider", options,
                            index=options.index(default) if default in options else len(options) - 1,
                            help="Providers appear when their API key is set in the environment / .env file.")
    model = st.text_input("Model (optional override)", value="")
    show_trace = st.toggle("Show tool trace", value=True)
    if st.button("🗑️ New conversation"):
        st.session_state.pop("agent", None)
        st.session_state.pop("history", None)
    st.caption("Knowledge base: TMDB 5000 (1,000 most-voted films). Answers are grounded in tool results.")

retriever = get_retriever()
config = (provider, model)
if st.session_state.get("config") != config or "agent" not in st.session_state:
    llm = None if provider == "offline" else make_client(provider, model or None)
    st.session_state.agent = MovieAgent(MovieTools(retriever), llm=llm)
    st.session_state.history = []
    st.session_state.config = config

# ------------------------------------------------------------------- chat
st.title("🎬 CineGuide")
st.caption("Ask for recommendations by mood, plot, genre, decade, actor — or 'movies like …'.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace") and show_trace:
            with st.expander("Tool trace"):
                st.json(msg["trace"])

if question := st.chat_input("e.g. A funny 90s comedy rated above 7"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = st.session_state.agent.ask(question)
                answer, trace = resp.answer, resp.tool_calls
                st.markdown(answer)
                st.caption(f"{resp.latency_s:.1f}s · {resp.steps} step(s) · sources: {', '.join(resp.sources[:6]) or '-'}")
            except Exception as exc:  # e.g. invalid API key or rate limit
                answer, trace = f"⚠️ LLM error: {exc}", None
                st.error(answer)
        if trace and show_trace:
            with st.expander("Tool trace"):
                st.json(trace)
    st.session_state.history.append({"role": "assistant", "content": answer, "trace": trace})
