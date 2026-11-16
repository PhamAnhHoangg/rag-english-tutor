import json
from types import SimpleNamespace

from eval.run_retrieval import evaluate, unique_refs
from tutor.chunking import Chunk
from tutor.retrieval import DenseRetriever, payload_to_chunk
from tutor.testset import generate, parse_reply, sample_chunks


class FakeQdrant:
    def __init__(self):
        self.calls = []

    def query_points(self, collection, query, limit, with_payload):
        self.calls.append((collection, query, limit))
        pts = [SimpleNamespace(score=0.9, payload={"text": "run: to move fast", "source": "wiktionary",
                                                   "ref": "wiktionary:run:verb", "sense": 0})]
        return SimpleNamespace(points=pts)


def test_dense_retriever_maps_payload_and_score():
    q = FakeQdrant()
    emb = SimpleNamespace(dim=2, embed_query=lambda text: [0.1, 0.2])
    r = DenseRetriever("unused", "english_tutor", emb, client=q)
    [c] = r.search("what does run mean", k=3)
    assert q.calls == [("english_tutor", [0.1, 0.2], 3)]
    assert c.ref == "wiktionary:run:verb" and c.meta == {"sense": 0, "score": 0.9}


def test_payload_to_chunk_keeps_extra_fields_as_meta():
    c = payload_to_chunk({"text": "t", "source": "s", "ref": "r", "start": 5})
    assert (c.text, c.meta) == ("t", {"start": 5})


class ListRetriever:
    def __init__(self, refs):
        self.refs = refs

    def search(self, q, k=5):
        return [Chunk("x", "s", r) for r in self.refs[q]][:k]


def test_evaluate_scores_and_dedupes_refs():
    qs = [{"id": "a", "source": "wiktionary", "question": "a", "gold_refs": ["w:run"]},
          {"id": "b", "source": "wikibooks", "question": "b", "gold_refs": ["b:tense"]}]
    r = ListRetriever({"a": ["w:go", "w:go", "w:run"], "b": ["x", "y"]})
    res = evaluate(r, qs, k=5)
    assert res["rows"][0]["top"] == ["w:go", "w:run"]
    assert res["recall@5"] == 0.5 and res["mrr"] == 0.25
    assert res["by_source"]["wiktionary"]["mrr"] == 0.5
    assert unique_refs([Chunk("x", "s", "a"), Chunk("y", "s", "a")]) == ["a"]


def test_parse_reply_handles_extra_text_and_bad_json():
    assert parse_reply('Sure! {"question": "Q?", "answer": "A."} hope it helps') == {"question": "Q?", "answer": "A."}
    assert parse_reply("no json here") is None
    assert parse_reply('{"question": "", "answer": "A"}') is None


def test_generate_uses_chunk_ref_as_gold_and_skips_bad_replies():
    chunks = [{"text": "affect: to influence", "source": "wiktionary", "ref": "wiktionary:affect:verb"},
              {"text": "broken", "source": "wiktionary", "ref": "wiktionary:x:noun"}]
    replies = iter([json.dumps({"question": "What does affect mean?", "answer": "To influence."}), "oops"])
    items = generate(chunks, SimpleNamespace(generate=lambda p: next(replies)))
    assert len(items) == 1 and items[0]["gold_refs"] == ["wiktionary:affect:verb"] and items[0]["reviewed"] is False


def test_sample_chunks_respects_counts_and_seed():
    chunks = [{"text": "x" * 50, "source": "tatoeba", "ref": str(i)} for i in range(20)]
    a = sample_chunks(chunks, {"tatoeba": 5, "wikibooks": 3}, seed=1)
    assert len(a) == 5 and a == sample_chunks(chunks, {"tatoeba": 5, "wikibooks": 3}, seed=1)


def test_indexer_and_retriever_round_trip_in_memory_qdrant():
    import pytest

    qdrant_client = pytest.importorskip("qdrant_client")
    from tutor.index import Indexer

    vecs = {"affect: to influence": [1.0, 0.0], "cat: an animal": [0.0, 1.0]}
    emb = SimpleNamespace(dim=2, embed_passages=lambda texts: [vecs[t] for t in texts],
                          embed_query=lambda q: [0.9, 0.1])
    client = qdrant_client.QdrantClient(":memory:")
    chunks = [Chunk(t, "wiktionary", f"wiktionary:{t.split(':')[0]}:x") for t in vecs]
    assert Indexer("unused", "t", emb, client=client).index(chunks) == 2
    assert DenseRetriever("unused", "t", emb, client=client).search("q", k=1)[0].ref == "wiktionary:affect:x"

    emb3 = SimpleNamespace(dim=3, embed_passages=lambda t: [], embed_query=lambda q: [])
    with pytest.raises(SystemExit, match="--recreate"):
        Indexer("unused", "t", emb3, client=client).index([])
    assert Indexer("unused", "t", emb3, client=client).index([], recreate=True) == 0


def test_make_embedder_rejects_unknown_backend():
    import pytest

    from tutor.embed import make_embedder

    with pytest.raises(ValueError, match="unknown"):
        make_embedder({"backend": "nope", "model": "x"})


def test_with_retry_retries_timeouts_then_gives_up():
    import httpx
    import pytest

    from tutor.index import with_retry

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ReadTimeout("timed out")
        return "ok"

    waits = []
    assert with_retry(flaky, sleep=waits.append) == "ok" and waits == [3.0, 6.0]

    def always():
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(SystemExit, match="docker compose ps"):
        with_retry(always, attempts=2, sleep=lambda s: None)


def test_create_retry_does_not_fail_if_first_create_landed():
    import httpx
    import pytest

    qdrant_client = pytest.importorskip("qdrant_client")
    from tutor.index import Indexer

    real = qdrant_client.QdrantClient(":memory:")

    class SlowCreate:
        """Creates the collection but reports a timeout, like a slow Docker volume."""
        def __getattr__(self, name):
            return getattr(real, name)

        def create_collection(self, *a, **kw):
            real.create_collection(*a, **kw)
            raise httpx.ReadTimeout("timed out")

    emb = SimpleNamespace(dim=2, embed_passages=lambda t: [], embed_query=lambda q: [])
    import tutor.index as idx
    orig = idx.time.sleep
    idx.time.sleep = lambda s: None
    try:
        Indexer("unused", "t", emb, client=SlowCreate()).ensure_collection()
    finally:
        idx.time.sleep = orig
    assert real.collection_exists("t")
