"""Retrievers. Each step on the experiment ladder adds one, and they stack:

  dense            embed the query, nearest chunks in Qdrant (baseline)
  bm25             keyword search over data/chunks.jsonl (no re-ingest needed)
  hybrid           dense + bm25, merged with reciprocal rank fusion (RRF)
  +rerank          a cross-encoder re-scores the top candidates
  split_comparisons  "X vs Y" questions retrieve X and Y separately, then interleave

`make_retriever(cfg)` builds the stack from configs/default.yaml.
"""
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tutor.chunking import Chunk


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...


def payload_to_chunk(payload: dict) -> Chunk:
    meta = {k: v for k, v in payload.items() if k not in {"text", "source", "ref"}}
    return Chunk(payload["text"], payload["source"], payload["ref"], meta)


def chunk_key(c: Chunk) -> tuple:
    """Identity of a chunk across retrievers (the same ref can hold several chunks)."""
    return c.ref, c.text


class DenseRetriever:
    """Baseline: embed the query and take the nearest chunks from Qdrant."""

    def __init__(self, qdrant_url: str, collection: str, embedder, client=None, timeout: int = 60):
        if client is None:
            from tutor.index import make_qdrant

            client = make_qdrant(qdrant_url, timeout)
        self.client, self.collection, self.embedder = client, collection, embedder

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        vec = self.embedder.embed_query(query)
        res = self.client.query_points(self.collection, query=vec, limit=k, with_payload=True)
        chunks = []
        for p in res.points:
            c = payload_to_chunk(p.payload)
            c.meta["score"] = p.score
            chunks.append(c)
        return chunks


# --- BM25 keyword search -------------------------------------------------------------

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read data/chunks.jsonl, which ingest writes next to every index build."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                chunks.append(Chunk(d["text"], d["source"], d["ref"], dict(d.get("meta") or {})))
    return chunks


class BM25Retriever:
    """Keyword search with BM25 (rank-bm25), built in memory from the chunks file.

    Dense embeddings blur exact words: "affect" and "effect" look alike, and a quoted
    sentence is not matched word for word. BM25 scores exact token overlap, weighted by
    how rare each token is, so it catches what the dense model misses. 22k chunks take a
    few seconds to index, so this needs no Qdrant change and no re-embedding.
    """

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        import numpy as np

        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(-scores)[:k]
        out = []
        for i in top:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            out.append(Chunk(c.text, c.source, c.ref, {**c.meta, "score": float(scores[i])}))
        return out


# --- Hybrid: reciprocal rank fusion -------------------------------------------------

def rrf(rankings: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank).

    It uses ranks, not raw scores, so cosine similarities and BM25 scores (different
    scales) can be merged without tuning weights. k=60 is the value from the original paper.
    """
    scores: dict[tuple, float] = {}
    first: dict[tuple, Chunk] = {}
    for ranking in rankings:
        for rank, c in enumerate(ranking, 1):
            key = chunk_key(c)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first.setdefault(key, c)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [Chunk(first[key].text, first[key].source, first[key].ref, {**first[key].meta, "score": scores[key]})
            for key in ordered]


class HybridRetriever:
    """Dense + BM25: take `depth` candidates from each and fuse them with RRF."""

    def __init__(self, retrievers: list[Retriever], depth: int = 50, rrf_k: int = 60):
        self.retrievers, self.depth, self.rrf_k = retrievers, depth, rrf_k

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        depth = max(self.depth, k)
        return rrf([r.search(query, k=depth) for r in self.retrievers], self.rrf_k)[:k]


# --- Cross-encoder reranking --------------------------------------------------------

class CrossEncoderScorer:
    """Scores (query, passage) pairs with a small ONNX cross-encoder from fastembed.

    A cross-encoder reads the query and passage together, so it judges relevance much
    better than comparing two separate embeddings, but it is too slow to run on all
    22k chunks. So it only re-scores the top candidates. MiniLM-L-6 is ~80 MB, CPU only.
    """

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from tutor.embed import default_cache_dir

        self.model = TextCrossEncoder(model_name=model, cache_dir=cache_dir or default_cache_dir())

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts, batch_size=32)]


class RerankRetriever:
    """Fetch `candidates` chunks from `base`, re-score them with `scorer`, keep the best k."""

    def __init__(self, base: Retriever, scorer: Callable[[str, list[str]], list[float]], candidates: int = 30):
        self.base, self.scorer, self.candidates = base, scorer, candidates

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        pool = self.base.search(query, k=max(self.candidates, k))
        if not pool:
            return []
        scores = self.scorer(query, [c.text for c in pool])
        ranked = sorted(zip(scores, range(len(pool))), key=lambda t: -t[0])
        return [Chunk(pool[i].text, pool[i].source, pool[i].ref, {**pool[i].meta, "rerank": s}) for s, i in ranked[:k]]


# --- "X vs Y" questions ---------------------------------------------------------------

_TERM = r"['\"]?([a-z][a-z' -]{0,30}?)['\"]?"
COMPARISON_PATTERNS = [
    re.compile(rf"difference(?:s)? between (?:the (?:words? )?)?(?:an? )?{_TERM} and (?:an? )?{_TERM}(?:\?|$| in | when | as )"),
    re.compile(rf"{_TERM}\s+(?:vs\.?|versus)\s+{_TERM}(?:\?|$| in | when )"),
    re.compile(rf"(?:should i use|do i use|do you use|when to use|to use) {_TERM} or {_TERM}(?:\?|$| in | when )"),
    re.compile(rf"(?:compare|confuse) {_TERM} (?:and|with) {_TERM}(?:\?|$)"),
]


def comparison_terms(query: str) -> list[str] | None:
    """The two terms of a comparison question ("affect vs effect"), or None."""
    q = " ".join(query.lower().split())
    for p in COMPARISON_PATTERNS:
        m = p.search(q)
        if m:
            a, b = (re.sub(r"['\"]? as an? \w+$", "", t).strip(" '\"") for t in m.groups())
            if a and b and a != b:
                return [a, b]
    return None


def interleave(lists: list[list[Chunk]]) -> list[Chunk]:
    """Round-robin merge without duplicates: a1, b1, c1, a2, b2, ..."""
    seen, out = set(), []
    for i in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if i < len(x) and chunk_key(x[i]) not in seen:
                seen.add(chunk_key(x[i]))
                out.append(x[i])
    return out


class HeadwordIndex:
    """Exact dictionary lookup: word -> its Wiktionary chunks, in sense order.

    When the question names the word, looking it up beats searching for it: BM25 and
    embeddings both struggle with very common words like "say" or "then".
    """

    def __init__(self, chunks: list[Chunk]):
        self.by_word: dict[str, list[Chunk]] = {}
        for c in chunks:
            if c.source == "wiktionary" and c.meta.get("word"):
                self.by_word.setdefault(c.meta["word"].lower(), []).append(c)

    def __call__(self, word: str) -> list[Chunk]:
        return self.by_word.get(word.lower(), [])


class ComparisonRetriever:
    """For "X vs Y" questions, fetch X and Y separately and interleave the results.

    One embedding of "difference between affect and effect" lands nearest whichever word
    dominates it, so only "affect" entries came back. Fetching each word on its own
    (dictionary lookup first, then a search for the word) gives both words a share of the
    top k. Other questions pass straight through.
    """

    def __init__(self, base: Retriever, lookup: Callable[[str], list[Chunk]] | None = None):
        self.base, self.lookup = base, lookup

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        terms = comparison_terms(query)
        if not terms:
            return self.base.search(query, k=k)
        per_term = []
        for t in terms:
            exact = self.lookup(t)[:2] if self.lookup else []
            per_term.append(interleave([exact, self.base.search(t, k=k)]))
        return interleave(per_term + [self.base.search(query, k=k)])[:k]


# --- Build from config ----------------------------------------------------------------

MODES = ("dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank")


def make_retriever(cfg: dict, mode: str | None = None, split_comparisons: bool | None = None) -> Retriever:
    """Build the retriever stack named by `mode` (default: retrieval.mode in the config)."""
    r = cfg["retrieval"]
    mode = mode or r.get("mode", "dense")
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}: use one of {', '.join(MODES)}")
    if split_comparisons is None:
        split_comparisons = r.get("split_comparisons", False)
    base_name = mode.split("+")[0]
    chunks_path = Path(r.get("chunks_path", "data/chunks.jsonl"))
    chunks: list[Chunk] = []
    if base_name != "dense" or split_comparisons:
        if not chunks_path.exists():
            raise SystemExit(f"{chunks_path} not found. BM25 and word lookup read the chunks file that "
                             "ingest writes; run scripts/ingest.py (it writes the file as it indexes).")
        chunks = load_chunks(chunks_path)

    def dense():
        from tutor.embed import make_embedder

        return DenseRetriever(r["qdrant_url"], r["collection"], make_embedder(cfg["embeddings"]))

    if base_name == "dense":
        base = dense()
    elif base_name == "bm25":
        base = BM25Retriever(chunks)
    else:
        base = HybridRetriever([dense(), BM25Retriever(chunks)],
                               depth=r.get("hybrid_depth", 50), rrf_k=r.get("rrf_k", 60))
    if mode.endswith("+rerank"):
        base = RerankRetriever(base, CrossEncoderScorer(r.get("reranker", "Xenova/ms-marco-MiniLM-L-6-v2")),
                               candidates=r.get("rerank_candidates", 30))
    if split_comparisons:
        base = ComparisonRetriever(base, HeadwordIndex(chunks))
    return base
