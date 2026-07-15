"""Test point_goal metrics computation."""

import pytest
from abotn_evaluator.point_goal.metrics import extract_metrics, aggregate


def _make_result(success=True, steps=50, travel=25.0, shortest=20.0,
                 path_coll=0, coll_len=0.0, total_dist=25.0):
    return {
        "success": success,
        "oracle_success": success,
        "steps": steps,
        "travel_length": travel,
        "shortest_path_length": shortest,
        "distance_to_goal": 0.3 if success else 5.0,
        "metrics": {
            "path_collision_count": path_coll,
            "collision_path_length": coll_len,
            "total_distance": total_dist,
            "is_path_collided": path_coll > 0,
            "initial_distance_to_goal": 20.0,
            "min_distance_to_goal": 0.3 if success else 3.0,
        },
    }


class TestExtractMetrics:
    def test_successful_no_collision(self):
        r = _make_result(success=True, path_coll=0)
        m = extract_metrics(r, collision_threshold=3)
        assert m["success"] == 1
        assert m["sr_new"] == 1
        assert m["tcr_new"] == pytest.approx(1.0)
        assert m["dcr_new"] == pytest.approx(1.0)

    def test_successful_with_collision_above_threshold(self):
        r = _make_result(success=True, path_coll=5)
        m = extract_metrics(r, collision_threshold=3)
        assert m["success"] == 1
        assert m["sr_new"] == 0
        assert m["spl_new"] == 0.0
        assert m["tcr_new"] == 0.0
        assert m["dcr_new"] == 0.0

    def test_indoor_zero_collision_threshold(self):
        r = _make_result(success=True, path_coll=1)
        m = extract_metrics(r, collision_threshold=1)
        assert m["success"] == 1
        assert m["sr_new"] == 0

        r0 = _make_result(success=True, path_coll=0)
        m0 = extract_metrics(r0, collision_threshold=1)
        assert m0["sr_new"] == 1

    def test_failed_episode(self):
        r = _make_result(success=False)
        m = extract_metrics(r, collision_threshold=3)
        assert m["success"] == 0
        assert m["spl_new"] == 0.0
        assert m["sr_new"] == 0

    def test_no_removed_keys(self):
        r = _make_result(success=True)
        m = extract_metrics(r, collision_threshold=3)
        for k in ("sr_point", "spl", "spl_point", "tcr", "dcr",
                  "point_collision_count", "is_point_collided"):
            assert k not in m


class TestAggregate:
    def test_all_successful(self):
        results = [_make_result(success=True) for _ in range(10)]
        metrics_list = [extract_metrics(r, 3) for r in results]
        agg = aggregate(metrics_list)
        assert agg["count"] == 10
        assert agg["success_rate"] == pytest.approx(1.0)
        assert agg["spl_new"] > 0

    def test_mixed_results(self):
        results = (
            [_make_result(success=True) for _ in range(5)]
            + [_make_result(success=False) for _ in range(5)]
        )
        metrics_list = [extract_metrics(r, 3) for r in results]
        agg = aggregate(metrics_list)
        assert agg["count"] == 10
        assert agg["success_rate"] == pytest.approx(0.5)

    def test_no_removed_keys(self):
        results = [_make_result(success=True) for _ in range(3)]
        metrics_list = [extract_metrics(r, 3) for r in results]
        agg = aggregate(metrics_list)
        for k in ("sr_point", "spl", "spl_point", "tcr", "tcr_global",
                  "tcr_new_global", "dcr", "dcr_global", "dcr_new_global",
                  "avg_point_collisions_per_task", "point_collision_task_rate"):
            assert k not in agg

    def test_empty_list(self):
        agg = aggregate([])
        assert agg["count"] == 0
