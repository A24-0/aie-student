from src.eval.metrics import (
    evaluate_run,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b", "x"], {"a", "b"}, 3) == 1.0
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(["x", "a", "b"], {"a"}, 3) == 0.5
    assert mrr_at_k(["a", "x"], {"a"}, 3) == 1.0
    assert mrr_at_k(["x", "y"], {"a"}, 3) == 0.0


def test_ndcg_is_one_when_relevant_first():
    assert ndcg_at_k(["a", "x"], {"a"}, 3) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 3) < 1.0


def test_evaluate_run_aggregates():
    preds = [(["a", "b"], {"a"}), (["x", "y"], {"y"})]
    m = evaluate_run(preds, k=2)
    assert m["hit@2"] == 1.0
    assert 0.0 < m["mrr@2"] <= 1.0
