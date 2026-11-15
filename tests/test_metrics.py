from eval.metrics import mrr, recall_at_k


def test_recall_and_mrr():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"b", "z"}, k=2) == 0.5
    assert mrr(retrieved, {"c"}) == 1 / 3
    assert mrr(retrieved, {"z"}) == 0.0
