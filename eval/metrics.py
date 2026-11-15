"""Retrieval metrics. `retrieved` is a ranked list of refs; `relevant` is the set of gold refs."""


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, ref in enumerate(retrieved, 1):
        if ref in relevant:
            return 1.0 / rank
    return 0.0
