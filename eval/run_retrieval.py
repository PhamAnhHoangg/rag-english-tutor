"""Score a retriever on the test set: recall@k and MRR, overall and per source."""
from collections import defaultdict

from eval.metrics import mrr, recall_at_k


def unique_refs(chunks) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        if c.ref not in seen:
            seen.add(c.ref)
            out.append(c.ref)
    return out


def evaluate(retriever, questions: list[dict], k: int = 5, depth: int = 20) -> dict:
    rows = []
    for q in questions:
        refs = unique_refs(retriever.search(q["question"], k=depth))
        gold = set(q["gold_refs"])
        rows.append({"id": q["id"], "source": q.get("source", "?"),
                     f"recall@{k}": recall_at_k(refs, gold, k), "mrr": mrr(refs, gold), "top": refs[:k]})

    def avg(rs, key):
        return round(sum(r[key] for r in rs) / len(rs), 4) if rs else 0.0

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    return {
        "n": len(rows),
        f"recall@{k}": avg(rows, f"recall@{k}"),
        "mrr": avg(rows, "mrr"),
        "by_source": {s: {"n": len(rs), f"recall@{k}": avg(rs, f"recall@{k}"), "mrr": avg(rs, "mrr")}
                      for s, rs in sorted(by_source.items())},
        "rows": rows,
    }
