import json
import subprocess
import sys
from pathlib import Path

from tutor.chunking import Chunk
from tutor.evaluation import evaluate, group_recall_at_k
from tutor.retrieval import (BM25Retriever, ComparisonRetriever, HeadwordIndex, HybridRetriever, RerankRetriever,
                             comparison_terms, interleave, make_retriever, rrf, tokenize)

ROOT = Path(__file__).resolve().parents[1]


def wk(word, sense=0, text=None):
    return Chunk(text or f"{word}: sense {sense} of {word}", "wiktionary", f"wiktionary:{word}:verb",
                 {"word": word, "sense": sense})


CHUNKS = [wk("affect"), wk("affect", 1), wk("effect"), wk("cat", text="cat: a small animal that says meow"),
          Chunk("I can't believe my eyes.\nWho is coming with me?", "tatoeba", "tatoeba:1#0")]


class ListRetriever:
    def __init__(self, chunks):
        self.chunks, self.queries = chunks, []

    def search(self, q, k=5):
        self.queries.append(q)
        return self.chunks[:k]


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("Can't believe MY eyes!") == ["can", "t", "believe", "my", "eyes"]


def test_bm25_finds_exact_sentence():
    r = BM25Retriever(CHUNKS)
    assert r.search("How do I say 'Who is coming with me?' naturally", k=1)[0].ref == "tatoeba:1#0"
    assert r.search("zzz", k=3) == []


def test_rrf_rewards_chunks_found_by_both():
    a, b, c = wk("a"), wk("b"), wk("c")
    fused = rrf([[a, b], [c, b]], k=60)
    assert fused[0].ref == b.ref and {x.ref for x in fused} == {a.ref, b.ref, c.ref}


def test_hybrid_asks_each_retriever_for_depth_and_cuts_to_k():
    d, s = ListRetriever([wk("a"), wk("b")]), ListRetriever([wk("b"), wk("c")])
    out = HybridRetriever([d, s], depth=10).search("q", k=2)
    assert [c.ref for c in out][0] == "wiktionary:b:verb" and len(out) == 2


def test_rerank_reorders_by_scorer():
    base = ListRetriever([wk("a"), wk("b"), wk("c")])
    scorer = lambda q, texts: [1.0 if "b:" in t else 0.0 for t in texts]  # noqa: E731
    out = RerankRetriever(base, scorer, candidates=3).search("q", k=2)
    assert out[0].ref == "wiktionary:b:verb" and out[0].meta["rerank"] == 1.0 and len(out) == 2


def test_comparison_terms():
    assert comparison_terms("What is the difference between affect and effect?") == ["affect", "effect"]
    assert comparison_terms("affect vs. effect") == ["affect", "effect"]
    assert comparison_terms("Should I use 'fewer' or 'less'?") == ["fewer", "less"]
    assert comparison_terms("What's the difference between 'finish' as a verb and 'finish' as a noun?") is None
    assert comparison_terms("When do I use the present perfect?") is None


def test_comparison_retriever_returns_both_words():
    # The base only ever finds "affect", like the dense baseline did.
    base = ListRetriever([wk("affect"), wk("affect", 1), wk("cat")])
    out = ComparisonRetriever(base, HeadwordIndex(CHUNKS)).search("What is the difference between affect and effect?", k=5)
    refs = [c.ref for c in out]
    assert "wiktionary:affect:verb" in refs[:2] and "wiktionary:effect:verb" in refs[:2]
    assert ComparisonRetriever(base).search("what is a cat", k=1)[0].ref == "wiktionary:affect:verb"


def test_interleave_dedupes():
    a, b = wk("a"), wk("b")
    assert [c.ref for c in interleave([[a, b], [b, a]])] == [a.ref, b.ref]


def test_group_recall_counts_each_word_once():
    groups = [["w:affect:noun", "w:affect:verb"], ["w:effect:noun"]]
    assert group_recall_at_k(["w:affect:verb", "w:x"], groups) == 0.5
    assert group_recall_at_k(["w:affect:verb", "w:effect:noun"], groups) == 1.0
    res = evaluate(ListRetriever([wk("affect")]), [{"id": "c1", "source": "compare", "question": "q",
                                                    "gold_groups": [["wiktionary:affect:verb"], ["wiktionary:effect:verb"]]}])
    assert res["recall@5"] == 0.5 and res["mrr"] == 1.0 and "ms_per_query" in res


def test_make_retriever_bm25_from_chunks_file(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps({"text": c.text, "source": c.source, "ref": c.ref, "meta": c.meta})
                              for c in CHUNKS), encoding="utf-8")
    cfg = {"retrieval": {"chunks_path": str(path), "split_comparisons": True}}
    r = make_retriever(cfg, mode="bm25")
    assert isinstance(r, ComparisonRetriever)
    refs = [c.ref for c in r.search("affect vs effect", k=3)]
    assert {"wiktionary:affect:verb", "wiktionary:effect:verb"} <= set(refs)


def test_scripts_run_without_pythonpath(tmp_path):
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    for script in ("eval_retrieval.py", "ask.py", "ingest.py", "make_testset.py"):
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
