"""Hybrid retrieval: TF-IDF (exact words) + Sentence-BERT (meaning).

TF-IDF is good at names and rare words ("Nolan", "wormhole"); embeddings are
good at paraphrases ("a film that makes you cry" -> drama / tearjerker). The
final score is a weighted mix of both, each min-max scaled to [0, 1].
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi - lo < 1e-12 else (scores - lo) / (hi - lo)


class HybridRetriever:
    """Search a DataFrame of movies by free-text query.

    Args:
        df: knowledge-base table with a ``doc`` column.
        mode: ``"hybrid"`` (default), ``"lexical"`` or ``"semantic"``.
        alpha: weight of the lexical score in hybrid mode (0..1).
        embedding_model: Sentence-Transformers model name.
    """

    def __init__(self, df: pd.DataFrame, mode: str = "hybrid", alpha: float = 0.4,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.df = df.reset_index(drop=True)
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(self.df["doc"])
        self.encoder = None
        self.embeddings = None
        self.mode = mode
        if mode in ("hybrid", "semantic"):
            try:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer(embedding_model)
                self.embeddings = self.encoder.encode(
                    self.df["doc"].tolist(), normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False)
            except Exception as exc:  # no internet / library missing -> degrade gracefully
                warnings.warn(f"Embeddings unavailable ({exc}); falling back to TF-IDF only.")
                self.mode = "lexical"

    # ------------------------------------------------------------------ scoring
    def lexical_scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return (self.tfidf @ q.T).toarray().ravel()  # cosine: rows are L2-normalised

    def semantic_scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        return self.embeddings @ q

    def scores(self, query: str) -> np.ndarray:
        if self.mode == "lexical":
            return self.lexical_scores(query)
        if self.mode == "semantic":
            return self.semantic_scores(query)
        return (self.alpha * _minmax(self.lexical_scores(query))
                + (1 - self.alpha) * _minmax(self.semantic_scores(query)))

    def similar_to_index(self, idx: int) -> np.ndarray:
        """Similarity of every movie to movie ``idx`` (used by 'find similar')."""
        lex = (self.tfidf @ self.tfidf[idx].T).toarray().ravel()
        if self.embeddings is None:
            return lex
        sem = self.embeddings @ self.embeddings[idx]
        return self.alpha * _minmax(lex) + (1 - self.alpha) * _minmax(sem)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 5, mask: np.ndarray | None = None) -> pd.DataFrame:
        """Return the top-``k`` rows with a ``score`` column.

        ``mask`` is an optional boolean array; rows where it is False are excluded
        (this is how the genre / year / rating filters are applied).
        """
        scores = self.scores(query).astype(float)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        order = [i for i in order if np.isfinite(scores[i])]
        out = self.df.iloc[order].copy()
        out["score"] = scores[order]
        return out
