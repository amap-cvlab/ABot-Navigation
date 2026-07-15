"""Metrics computation for POI-goal navigation evaluation.

Extends the point-goal metrics module with:

* SPL formula that subtracts ``arrive_threshold`` from shortest path length.
* Per-POI breakdown in aggregate statistics.
* Collision termination statistics specific to the POI-goal task.

All metric formulas match the POI-goal evaluator logic in
``GaussianEvaluatorPoiGoal``.
"""

import csv
import glob
import json
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from abotn_evaluator.point_goal.metrics import (
    INDOOR_EASY_SCENES,
    INDOOR_HARD_SCENES,
    OUTDOOR_EXCLUDED_SCENES,
    classify_indoor,
    classify_outdoor,
    load_all_results,
    print_summary_table,
    _print_group_metrics,
)


# ============================================================================
# Internal helpers
# ============================================================================

def _compute_poi_spl(
    success: bool,
    travel_length: float,
    shortest_path_length: float,
    arrive_threshold: float,
) -> float:
    """Compute SPL with arrive_threshold subtracted from shortest path.

    ``effective_shortest = max(shortest - arrive_threshold, 1e-3)``
    ``SPL = success * effective_shortest / max(travel, effective_shortest)``
    """
    if not success:
        return 0.0
    if shortest_path_length <= 0:
        return 0.0
    effective_shortest = max(
        shortest_path_length - max(arrive_threshold, 0.0), 1e-3
    )
    return effective_shortest / max(travel_length, effective_shortest)


# ============================================================================
# Public API -- per-task extraction
# ============================================================================

def extract_metrics(
    r: dict,
    collision_threshold: int = 3,
    arrive_threshold: float = 2.0,
) -> OrderedDict:
    """Extract metrics from a single POI-goal ``result.json`` dictionary.

    Parameters
    ----------
    r : dict
        Parsed contents of a ``result.json`` file.
    collision_threshold : int
        Collision count threshold for computing SR_NEW / SR_POINT.
    arrive_threshold : float
        The arrival distance threshold used during evaluation.  Used to
        recompute the adjusted SPL.

    Returns
    -------
    OrderedDict
        Flat dictionary of metric name -> value.
    """
    m = r.get("metrics", {}) or {}
    out = OrderedDict()

    success = bool(r.get("success", False))
    out["success"] = int(success)
    out["oracle_success"] = int(bool(r.get("oracle_success", False)))

    # POI name
    out["poi_name"] = m.get("poi_name", r.get("target_label", "unknown"))

    # Use the threshold stored in the result if available, else fallback
    task_threshold = float(
        m.get("pointgoal_arrive_threshold", arrive_threshold)
    )

    # SPL (with arrive_threshold adjustment)
    travel = float(r.get("travel_length", 0.0))
    sp = float(r.get("shortest_path_length", 0.0))
    out["spl"] = _compute_poi_spl(success, travel, sp, task_threshold)

    # Collision-related metrics
    collision_mode = m.get("collision_mode", "off")
    collision_count = int(m.get("collision_count", 0))
    collision_terminated = bool(m.get("collision_terminated", False))

    out["collision_mode"] = collision_mode
    out["collision_count"] = collision_count
    out["collision_rate"] = float(m.get("collision_rate", 0.0))
    out["max_consecutive_collision"] = int(
        m.get("max_consecutive_collision", 0)
    )
    out["has_collision"] = int(bool(m.get("has_collision", False)))
    out["collision_terminated"] = int(collision_terminated)
    out["collision_skipped_near_goal"] = int(
        m.get("collision_skipped_near_goal", 0)
    )

    # Path/point collision counts (for backward compat with point_goal metrics)
    out["point_collision_count"] = int(m.get("point_collision_count", 0))
    out["path_collision_count"] = int(m.get("path_collision_count", 0))
    out["collision_path_length"] = float(m.get("collision_path_length", 0.0))
    out["total_distance"] = float(m.get("total_distance", 0.0))
    out["is_point_collided"] = int(bool(m.get("is_point_collided", False)))
    out["is_path_collided"] = int(bool(m.get("is_path_collided", False)))

    # General metrics
    out["steps"] = int(r.get("steps", 0))
    out["travel_length"] = travel
    out["shortest_path_length"] = sp
    out["initial_distance_to_goal"] = float(
        m.get("initial_distance_to_goal", 0.0)
    )
    # For error tasks, metrics dict is empty and distance_to_goal is inf;
    # fall back to 0.0 to prevent inf from polluting aggregate statistics.
    _raw_min = m.get("min_distance_to_goal", 0.0)
    _raw_final = r.get("distance_to_goal", 0.0)
    out["min_distance_to_goal"] = float(_raw_min) if np.isfinite(_raw_min) else 0.0
    out["final_distance_to_goal"] = float(_raw_final) if np.isfinite(_raw_final) else 0.0

    # Status
    out["status"] = r.get("status", "unknown")

    # Sanity check flags
    out["start_in_obstacle"] = int(bool(m.get("start_in_obstacle", False)))
    out["goal_in_obstacle"] = int(bool(m.get("goal_in_obstacle", False)))

    return out


# ============================================================================
# Public API -- aggregation
# ============================================================================

def aggregate(
    metrics_list: List[OrderedDict],
    arrive_threshold: float = 2.0,
) -> dict:
    """Aggregate a list of per-task metric dicts into summary statistics.

    Parameters
    ----------
    metrics_list : list of OrderedDict
        Each element is the return value of :func:`extract_metrics`.
    arrive_threshold : float
        The arrival threshold (for display/context only; SPL is already
        computed per-task).

    Returns
    -------
    dict
        Aggregated metrics including per-POI breakdown and collision stats.
    """
    n = len(metrics_list)
    if n == 0:
        return {"count": 0}

    def mean(key):
        vals = [m.get(key, 0.0) for m in metrics_list]
        finite_vals = [v for v in vals if np.isfinite(v)]
        if not finite_vals:
            return 0.0
        return float(np.mean(finite_vals))

    def total(key):
        return float(np.sum([m.get(key, 0.0) for m in metrics_list]))

    # Core metrics
    agg = {
        "count": n,
        "arrive_threshold": arrive_threshold,
        "success_rate": mean("success"),
        "oracle_success_rate": mean("oracle_success"),
        "spl": mean("spl"),
        "avg_steps": mean("steps"),
        "avg_travel_length": mean("travel_length"),
        "avg_shortest_path_length": mean("shortest_path_length"),
        "avg_initial_distance": mean("initial_distance_to_goal"),
        "avg_min_distance": mean("min_distance_to_goal"),
        "avg_final_distance": mean("final_distance_to_goal"),
    }

    # Collision aggregate
    collision_enabled = [
        m for m in metrics_list
        if m.get("collision_mode", "off") != "off"
    ]
    if collision_enabled:
        n_coll = len(collision_enabled)
        tasks_with_coll = sum(
            1 for m in collision_enabled if m.get("has_collision", 0)
        )
        total_coll_steps = sum(
            m.get("collision_count", 0) for m in collision_enabled
        )
        terminated_count = sum(
            1 for m in collision_enabled if m.get("collision_terminated", 0)
        )
        agg["collision"] = {
            "tasks_evaluated": n_coll,
            "tasks_with_collision": tasks_with_coll,
            "collision_task_rate": tasks_with_coll / n_coll,
            "total_collision_steps": int(total_coll_steps),
            "avg_collision_per_task": total_coll_steps / n_coll,
            "avg_collision_rate": float(np.mean([
                m.get("collision_rate", 0.0) for m in collision_enabled
            ])),
            "collision_terminated_count": terminated_count,
        }
    else:
        agg["collision"] = {"tasks_evaluated": 0}

    # Status distribution
    status_dist: Dict[str, int] = {}
    for m in metrics_list:
        s = m.get("status", "unknown")
        status_dist[s] = status_dist.get(s, 0) + 1
    agg["status_distribution"] = status_dist

    # Per-POI breakdown
    poi_groups: Dict[str, List[OrderedDict]] = defaultdict(list)
    for m in metrics_list:
        poi = m.get("poi_name", "unknown")
        poi_groups[poi].append(m)

    poi_stats: Dict[str, Dict[str, Any]] = {}
    for poi, group in poi_groups.items():
        g_n = len(group)
        g_success = sum(m.get("success", 0) for m in group)
        g_spl = float(np.mean([m.get("spl", 0.0) for m in group]))
        g_collision = sum(m.get("has_collision", 0) for m in group)
        g_terminated = sum(
            m.get("collision_terminated", 0) for m in group
        )
        poi_stats[poi] = {
            "total": g_n,
            "success": int(g_success),
            "success_rate": g_success / g_n if g_n > 0 else 0.0,
            "avg_spl": g_spl,
            "collision_count": int(g_collision),
            "collision_terminated_count": int(g_terminated),
        }
    agg["poi_stats"] = poi_stats

    return agg


# ============================================================================
# Public API -- full pipeline
# ============================================================================

def analyze_and_report(
    result_dir: str,
    mode: str = "indoor",
    collision_threshold: int = 3,
    arrive_threshold: float = 2.0,
    min_distance: float = 5.0,
    max_distance: float = 50.0,
    exclude_scenes: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    per_scene: bool = False,
    per_poi: bool = True,
) -> dict:
    """Run the full POI-goal metrics analysis pipeline.

    1. Load all ``result.json`` files from *result_dir*.
    2. Exclude specified scenes.
    3. Classify by difficulty (indoor easy/hard or outdoor short/medium/long).
    4. Compute and aggregate metrics per group and overall.
    5. Print summary tables and per-POI breakdowns.
    6. Save JSON output.

    Parameters
    ----------
    result_dir : str
        Root evaluation output directory.
    mode : str
        ``"outdoor"`` or ``"indoor"``.
    collision_threshold : int
        Collision-count threshold for legacy SR_NEW (kept for compat).
    arrive_threshold : float
        Arrival distance threshold used during evaluation.
    min_distance, max_distance : float
        Distance range for outdoor difficulty classification.
    exclude_scenes : list of str, optional
        Scene IDs to exclude.
    output_path : str, optional
        Path for JSON output.  Defaults to
        ``{result_dir}/poi_goal_analysis.json``.
    per_scene : bool
        If True, print per-scene breakdown.
    per_poi : bool
        If True, print per-POI breakdown (default True).

    Returns
    -------
    dict
        Full analysis payload (also saved as JSON).
    """
    results = load_all_results(result_dir)
    if not results:
        print("[ERROR] No result.json files found")
        return {"count": 0}

    # Default exclusions
    if exclude_scenes is None:
        if mode == "outdoor":
            exclude_scenes = list(OUTDOOR_EXCLUDED_SCENES)
        else:
            exclude_scenes = []

    if exclude_scenes:
        before = len(results)
        results = [
            r for r in results
            if r.get("episode_id", "") not in exclude_scenes
        ]
        print(
            f"Excluded scenes {exclude_scenes}: "
            f"filtered {before - len(results)} tasks, "
            f"{len(results)} remaining"
        )
        if not results:
            print("[ERROR] No data remaining after exclusion")
            return {"count": 0}

    # Classify
    if mode == "outdoor":
        groups = classify_outdoor(results, min_distance, max_distance)
    else:
        groups = classify_indoor(results)

    # Aggregate per group
    groups_agg: Dict[str, dict] = OrderedDict()
    for label, group_results in groups.items():
        if not group_results:
            groups_agg[label] = {"count": 0}
            continue
        metrics_list = [
            extract_metrics(r, collision_threshold, arrive_threshold)
            for r in group_results
        ]
        groups_agg[label] = aggregate(metrics_list, arrive_threshold)

    # Overall
    all_metrics = [
        extract_metrics(r, collision_threshold, arrive_threshold)
        for r in results
    ]
    overall_agg = aggregate(all_metrics, arrive_threshold)

    # Print tables
    print(f"\n{'=' * 60}")
    print(f"POI Goal Metrics Analysis (arrive_threshold={arrive_threshold}m)")
    print(f"{'=' * 60}")

    # Per-group detailed print
    for label, agg in groups_agg.items():
        _print_group_metrics(label, agg)
    _print_group_metrics("overall", overall_agg)

    # Per-POI breakdown
    if per_poi:
        _print_per_poi_metrics(overall_agg)

    # Collision termination stats
    coll = overall_agg.get("collision", {})
    if coll.get("tasks_evaluated", 0) > 0:
        print(f"\nCollision Summary:")
        print(
            f"  Tasks with collision: "
            f"{coll['tasks_with_collision']}/{coll['tasks_evaluated']} "
            f"({coll['collision_task_rate'] * 100:.1f}%)"
        )
        print(
            f"  Avg collision steps/task: "
            f"{coll['avg_collision_per_task']:.2f}"
        )
        print(
            f"  Collision terminated: "
            f"{coll['collision_terminated_count']} tasks"
        )

    # Per-scene breakdown
    if per_scene:
        _print_poi_per_scene_metrics(results, collision_threshold, arrive_threshold)

    # Save JSON
    out_path = output_path or os.path.join(
        result_dir, "poi_goal_analysis.json"
    )
    payload: Dict[str, Any] = {
        "result_dir": os.path.abspath(result_dir),
        "task_type": "poi_goal",
        "mode": mode,
        "arrive_threshold": arrive_threshold,
        "collision_threshold": collision_threshold,
        "total_count": len(results),
    }
    if mode == "outdoor":
        payload["distance_config"] = {
            "min_distance": min_distance,
            "max_distance": max_distance,
            "step": (max_distance - min_distance) / 3.0,
        }
    payload["groups"] = {label: agg for label, agg in groups_agg.items()}
    payload["overall"] = overall_agg

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved POI goal analysis JSON: {out_path}")

    return payload


# ============================================================================
# Internal printing helpers
# ============================================================================

def _print_poi_per_scene_metrics(
    results: List[dict],
    collision_threshold: int,
    arrive_threshold: float,
) -> None:
    scene_groups: Dict[str, list] = defaultdict(list)
    for r in results:
        scene_groups[r.get("episode_id", "unknown")].append(r)

    print(f"\n{'=' * 80}")
    print("Per-scene metrics (POI-Goal)")
    print(f"{'=' * 80}")

    col_w = 10
    header = (
        f"{'scene':<20}{'count':>{col_w}}{'SR%':>{col_w}}"
        f"{'spl':>{col_w}}{'coll':>{col_w}}{'term':>{col_w}}{'avg_sp':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for scene_name in sorted(scene_groups):
        mlist = [
            extract_metrics(r, collision_threshold, arrive_threshold)
            for r in scene_groups[scene_name]
        ]
        agg = aggregate(mlist, arrive_threshold)
        n = agg.get("count", 0)
        coll = agg.get("collision", {})
        print(
            f"{scene_name:<20}{n:>{col_w}d}"
            f"{agg.get('success_rate', 0):>{col_w}.4f}"
            f"{agg.get('spl', 0):>{col_w}.4f}"
            f"{coll.get('tasks_with_collision', 0):>{col_w}d}"
            f"{coll.get('collision_terminated_count', 0):>{col_w}d}"
            f"{agg.get('avg_shortest_path_length', 0):>{col_w}.1f}"
        )


def _print_per_poi_metrics(overall_agg: dict) -> None:
    """Print per-POI breakdown table."""
    poi_stats = overall_agg.get("poi_stats", {})
    if not poi_stats:
        return

    print(f"\n{'=' * 60}")
    print("Per-POI Breakdown")
    print(f"{'=' * 60}")

    col_w = 10
    header = (
        f"{'POI':<30}{'count':>{col_w}}{'success':>{col_w}}"
        f"{'SR%':>{col_w}}{'avg_spl':>{col_w}}"
        f"{'coll':>{col_w}}{'term':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for poi in sorted(poi_stats.keys()):
        stat = poi_stats[poi]
        # Truncate long POI names for display
        display_name = poi[:28] if len(poi) > 28 else poi
        print(
            f"{display_name:<30}"
            f"{stat['total']:>{col_w}d}"
            f"{stat['success']:>{col_w}d}"
            f"{stat['success_rate'] * 100:>{col_w}.1f}"
            f"{stat['avg_spl']:>{col_w}.4f}"
            f"{stat['collision_count']:>{col_w}d}"
            f"{stat['collision_terminated_count']:>{col_w}d}"
        )
    print(f"{'=' * 60}")


def _cli_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Aggregate POI-goal evaluation metrics. "
                    "Can be run at any time — even while evaluation is still in progress — "
                    "to get interim results from completed tasks.",
    )
    parser.add_argument("--result-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--mode", default="indoor", choices=["outdoor", "indoor"])
    parser.add_argument("--collision-threshold", type=int, default=3)
    parser.add_argument("--arrive-threshold", type=float, default=2.0)
    parser.add_argument("--min-distance", type=float, default=5.0)
    parser.add_argument("--max-distance", type=float, default=50.0)
    parser.add_argument("--exclude-scenes", nargs="*", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--per-scene", action="store_true")
    parser.add_argument("--per-poi", action="store_true", default=True)
    parser.add_argument("--no-per-poi", action="store_false", dest="per_poi")
    args = parser.parse_args(argv)
    analyze_and_report(
        result_dir=args.result_dir,
        mode=args.mode,
        collision_threshold=args.collision_threshold,
        arrive_threshold=args.arrive_threshold,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        exclude_scenes=args.exclude_scenes,
        output_path=args.output_path,
        per_scene=args.per_scene,
        per_poi=args.per_poi,
    )


if __name__ == "__main__":
    _cli_main()
