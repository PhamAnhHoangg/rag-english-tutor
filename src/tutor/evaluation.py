"""Retrieval metrics and the eval loop.

This lives inside the `tutor` package (not in eval/) so scripts can import it without
setting PYTHONPATH. eval/ only holds data: test questions and saved results.
"""
import time
from collections import defaultdict


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Share of the gold refs found in the top k. `retrieved` is a ranked list of refs."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first gold ref, 0 if none was retrieved."""
    for rank, ref in enumerate(retrieved, 1):
        if ref in relevant:
            return 1.0 / rank
    return 0.0


def group_recall_at_k(retrieved: list[str], groups: list[list[str]], k: int = 5) -> float:
    """Share of groups with at least one ref in the top k.

    For "X vs Y" questions each word is a group (its noun and verb entries both count),
    so 1.0 means both words were retrieved and 0.5 means only one was.
    """
    if not groups:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for g in groups if top & set(g)) / len(groups)


def unique_refs(chunks) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        if c.ref not in seen:
            seen.add(c.ref)
            out.append(c.ref)
    return out


def evaluate(retriever, questions: list[dict], k: int = 5, depth: int = 20) -> dict:
    """Score a retriever: recall@k and MRR, overall and per source, plus mean latency."""
    rows = []
    for q in questions:
        t = time.perf_counter()
        refs = unique_refs(retriever.search(q["question"], k=depth))
        ms = (time.perf_counter() - t) * 1000
        if "gold_groups" in q:
            gold = {r for g in q["gold_groups"] for r in g}
            recall = group_recall_at_k(refs, q["gold_groups"], k)
        else:
            gold = set(q["gold_refs"])
            recall = recall_at_k(refs, gold, k)
        rows.append({"id": q["id"], "source": q.get("source", "?"),
                     f"recall@{k}": recall, "mrr": mrr(refs, gold),
                     "ms": round(ms, 1), "top": refs[:k]})

    def avg(rs, key):
        return round(sum(r[key] for r in rs) / len(rs), 4) if rs else 0.0

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    return {
        "n": len(rows),
        f"recall@{k}": avg(rows, f"recall@{k}"),
        "mrr": avg(rows, "mrr"),
        "ms_per_query": round(avg(rows, "ms"), 1),
        "by_source": {s: {"n": len(rs), f"recall@{k}": avg(rs, f"recall@{k}"), "mrr": avg(rs, "mrr")}
                      for s, rs in sorted(by_source.items())},
        "rows": rows,
    }
