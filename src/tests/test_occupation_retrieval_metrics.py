from src.occupation_retrieval.metrics import (
    compute_candidate_accuracy,
    reciprocal_rank,
    same_hierarchy_level,
    summarize_model_metrics,
)


def test_compute_candidate_accuracy_handles_hit_miss_and_empty_values() -> None:
    assert compute_candidate_accuracy("A", "A") == 1
    assert compute_candidate_accuracy("a", "A") == 1
    assert compute_candidate_accuracy("B", "A") == 0
    assert compute_candidate_accuracy(None, "A") == 0
    assert compute_candidate_accuracy("A", None) == 0


def test_reciprocal_rank_handles_rank_and_miss() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(None) == 0.0
    assert reciprocal_rank(0) == 0.0


def test_same_hierarchy_level_compares_mapped_values() -> None:
    mapping = {
        "1-01-01": "1-01",
        "1-01-02": "1-01",
        "1-02-01": "1-02",
    }
    assert same_hierarchy_level("1-01-01", "1-01-02", mapping)
    assert not same_hierarchy_level("1-01-01", "1-02-01", mapping)
    assert not same_hierarchy_level("missing", "1-01-01", mapping)


def test_summarize_model_metrics_returns_zero_for_empty_denominators() -> None:
    summary = summarize_model_metrics({})
    assert summary["candidate_acc"] == 0.0
    assert summary["mean_human_rank"] == 0.0
    assert summary["mrr"] == 0.0
    assert summary["subclass_acc"] == 0.0
    assert summary["midclass_acc"] == 0.0
    assert summary["major_acc"] == 0.0
    assert summary["ds_side_human_pct"] == 0.0
    assert summary["ds_side_ds_pct"] == 0.0


def test_summarize_model_metrics_computes_percentages_and_means() -> None:
    summary = summarize_model_metrics(
        {
            "candidate_hit": 2,
            "candidate_total": 4,
            "human_rank_in_candidates": [1, 3],
            "reciprocal_ranks": [1.0, 0.5, 0.25],
            "subclass_hit": 3,
            "midclass_hit": 2,
            "major_hit": 1,
            "full_total": 4,
            "ds_side_human": 3,
            "ds_side_ds": 1,
            "ds_total": 5,
        }
    )
    assert summary["candidate_acc"] == 50.0
    assert summary["mean_human_rank"] == 2.0
    assert round(summary["mrr"], 4) == 0.5833
    assert summary["subclass_acc"] == 75.0
    assert summary["midclass_acc"] == 50.0
    assert summary["major_acc"] == 25.0
    assert summary["ds_side_human_pct"] == 60.0
    assert summary["ds_side_ds_pct"] == 20.0
