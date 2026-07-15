"""Metrics computation for point-goal navigation evaluation.

Provides functions to extract per-task metrics from ``result.json`` files,
aggregate across episodes, classify by difficulty, and produce summary reports.

All metric formulas match the original ``analyze_difficulty_results.py``
implementation exactly.
"""

import glob
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# Scene classification constants
# ============================================================================

OUTDOOR_EXCLUDED_SCENES = ["park3"]

INDOOR_EASY_SCENES = [
    "0802_840243", "0803_840265", "0821_841631", "0822_841630",
    "0827_841619", "0841_841759", "0845_841765", "0854_841775",
]

INDOOR_HARD_SCENES = [
    "0813_841249", "0814_841252", "0832_840249", "0833_840508",
    "0837_841153", "0843_841761", "0847_841768", "0861_841783",
]


# ============================================================================
# Internal helpers
# ============================================================================

def _is_success_new(r: dict, collision_threshold: int) -> bool:
    """SR_NEW: success AND path_collision_count < threshold."""
    return (
        r.get("success", False)
        and r.get("metrics", {}).get("path_collision_count", 0) < collision_threshold
    )


def _spl_new(r: dict, collision_threshold: int) -> float:
    """SPL_NEW: SPL based on SR_NEW (path-collision-aware success)."""
    if not _is_success_new(r, collision_threshold):
        return 0.0
    travel = r.get("travel_length", 0.0)
    sp = r.get("shortest_path_length", 0.0)
    if sp <= 0:
        return 0.0
    return max(0.0, sp / max(travel, sp))


# ============================================================================
# Public API -- per-task extraction
# ============================================================================

def extract_metrics(r: dict, collision_threshold: int = 3) -> OrderedDict:
    """Extract metrics from a single ``result.json`` dictionary.

    Parameters
    ----------
    r : dict
        Parsed contents of a ``result.json`` file.
    collision_threshold : int
        Collision count threshold for SR_NEW / SR_POINT.

    Returns
    -------
    OrderedDict
        Flat dictionary of metric name -> value.
    """
    m = r.get("metrics", {}) or {}
    out = OrderedDict()
    out["success"] = int(bool(r.get("success", False)))
    out["oracle_success"] = int(bool(r.get("oracle_success", False)))
    out["sr_new"] = int(_is_success_new(r, collision_threshold))
    out["spl_new"] = float(_spl_new(r, collision_threshold))

    # SR_NEW condition shared by TCR_NEW / DCR_NEW
    is_sn = _is_success_new(r, collision_threshold)

    # TCR_NEW (per-task): gated by SR_NEW, based on path_collision_count
    steps = int(r.get("steps", 0))
    path_coll = int(m.get("path_collision_count", 0))
    if is_sn and steps > 0:
        out["tcr_new"] = (steps - path_coll) / steps
    else:
        out["tcr_new"] = 0.0

    # DCR_NEW (per-task): gated by SR_NEW condition
    total_dist = float(m.get("total_distance", 0.0))
    coll_len = float(m.get("collision_path_length", 0.0))
    if is_sn and total_dist > 0:
        out["dcr_new"] = (total_dist - coll_len) / total_dist
    else:
        out["dcr_new"] = 0.0

    out["steps"] = int(r.get("steps", 0))
    out["travel_length"] = float(r.get("travel_length", 0.0))
    out["shortest_path_length"] = float(r.get("shortest_path_length", 0.0))
    out["initial_distance_to_goal"] = float(
        m.get("initial_distance_to_goal", 0.0)
    )
    out["min_distance_to_goal"] = float(
        m.get("min_distance_to_goal", r.get("distance_to_goal", 0.0))
    )
    out["final_distance_to_goal"] = float(r.get("distance_to_goal", 0.0))
    out["path_collision_count"] = int(m.get("path_collision_count", 0))
    out["collision_path_length"] = float(m.get("collision_path_length", 0.0))
    out["total_distance"] = float(m.get("total_distance", 0.0))
    out["is_path_collided"] = int(bool(m.get("is_path_collided", False)))
    return out


# ============================================================================
# Public API -- aggregation
# ============================================================================

def aggregate(metrics_list: List[OrderedDict]) -> dict:
    """Aggregate a list of per-task metric dicts into summary statistics.

    Parameters
    ----------
    metrics_list : list of OrderedDict
        Each element is the return value of :func:`extract_metrics`.

    Returns
    -------
    dict
        Aggregated metrics including per-task means and global ratios.
    """
    n = len(metrics_list)
    if n == 0:
        return {"count": 0}

    def mean(key):
        return float(np.mean([m[key] for m in metrics_list]))

    def total(key):
        return float(np.sum([m[key] for m in metrics_list]))

    total_dist = total("total_distance")
    total_coll_len = total("collision_path_length")

    return {
        "count": n,
        "success_rate": mean("success"),
        "oracle_success_rate": mean("oracle_success"),
        "sr_new": mean("sr_new"),
        "spl_new": mean("spl_new"),
        "tcr_new": mean("tcr_new"),
        "dcr_new": mean("dcr_new"),
        "avg_steps": mean("steps"),
        "avg_travel_length": mean("travel_length"),
        "avg_shortest_path_length": mean("shortest_path_length"),
        "avg_initial_distance": mean("initial_distance_to_goal"),
        "avg_min_distance": mean("min_distance_to_goal"),
        "avg_final_distance": mean("final_distance_to_goal"),
        "avg_path_collisions_per_task": mean("path_collision_count"),
        "avg_collision_meters_per_task": mean("collision_path_length"),
        "global_collision_path_ratio": (
            total_coll_len / total_dist if total_dist > 0 else 0.0
        ),
        "path_collision_task_rate": mean("is_path_collided"),
    }


# ============================================================================
# Public API -- difficulty classification
# ============================================================================

def classify_outdoor(
    results: List[dict],
    min_dist: float = 5.0,
    max_dist: float = 50.0,
) -> Dict[str, List[dict]]:
    """Split outdoor results into short / medium / long difficulty buckets.

    Buckets are determined by ``shortest_path_length``:

    * short:  [min_dist, min_dist + step)
    * medium: [min_dist + step, min_dist + 2*step)
    * long:   [min_dist + 2*step, max_dist * 1.5]

    where ``step = (max_dist - min_dist) / 3``.

    Parameters
    ----------
    results : list of dict
        Raw ``result.json`` dicts.
    min_dist, max_dist : float
        Distance range for bucketing.

    Returns
    -------
    OrderedDict
        Keys ``"short"``, ``"medium"``, ``"long"``, ``"out_of_range"``,
        each mapping to a list of result dicts.
    """
    step = (max_dist - min_dist) / 3.0
    boundaries = [
        ("short", min_dist, min_dist + step),
        ("medium", min_dist + step, min_dist + 2 * step),
        ("long", min_dist + 2 * step, max_dist),
    ]

    groups: Dict[str, List[dict]] = OrderedDict()
    for label, _, _ in boundaries:
        groups[label] = []
    groups["out_of_range"] = []

    for r in results:
        sp = r.get("shortest_path_length", 0.0)
        placed = False
        for label, lo, hi in boundaries:
            if label == "long":
                if lo <= sp <= hi * 1.5:
                    groups[label].append(r)
                    placed = True
                    break
            else:
                if lo <= sp < hi:
                    groups[label].append(r)
                    placed = True
                    break
        if not placed:
            groups["out_of_range"].append(r)

    print(
        f"\nOutdoor difficulty distribution "
        f"(range [{min_dist}, {max_dist}]m, step={step:.1f}m):"
    )
    for label, lo, hi in boundaries:
        print(f"  {label:>8}: [{lo:.1f}, {hi:.1f}) -> {len(groups[label])} tasks")
    if groups["out_of_range"]:
        print(f"  {'out_of_range':>8}: {len(groups['out_of_range'])} tasks")

    return groups


def classify_indoor(results: List[dict]) -> Dict[str, List[dict]]:
    """Split indoor results into easy / hard difficulty buckets by scene name.

    Parameters
    ----------
    results : list of dict
        Raw ``result.json`` dicts.

    Returns
    -------
    OrderedDict
        Keys ``"easy"``, ``"hard"``, ``"unknown"``.
    """
    groups: Dict[str, List[dict]] = OrderedDict()
    groups["easy"] = []
    groups["hard"] = []
    groups["unknown"] = []

    for r in results:
        ep = r.get("episode_id", "")
        if ep in INDOOR_EASY_SCENES:
            groups["easy"].append(r)
        elif ep in INDOOR_HARD_SCENES:
            groups["hard"].append(r)
        else:
            groups["unknown"].append(r)

    print("\nIndoor difficulty distribution:")
    print(
        f"  easy: {len(groups['easy'])} tasks "
        f"(scenes: {sorted(set(r['episode_id'] for r in groups['easy']))})"
    )
    print(
        f"  hard: {len(groups['hard'])} tasks "
        f"(scenes: {sorted(set(r['episode_id'] for r in groups['hard']))})"
    )
    if groups["unknown"]:
        print(f"  unknown: {len(groups['unknown'])} tasks")

    return groups


# ============================================================================
# Public API -- loading results
# ============================================================================

def load_all_results(result_dir: str) -> List[dict]:
    """Recursively load all ``result.json`` files under *result_dir*.

    Parameters
    ----------
    result_dir : str
        Root directory to search.

    Returns
    -------
    list of dict
        Parsed result dicts, each augmented with a ``__path__`` key.
    """
    pattern = os.path.join(result_dir, "**", "result.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    results = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to load {p}: {exc}")
            continue
        ep = data.get("episode_id")
        tid = data.get("task_id")
        if ep is None or tid is None:
            print(f"[WARN] Missing episode_id or task_id: {p}")
            continue
        data["__path__"] = p
        results.append(data)
    print(f"[{result_dir}] Loaded {len(results)} result.json files")
    return results


# ============================================================================
# Public API -- printing / CSV
# ============================================================================

def print_summary_table(
    groups_agg: Dict[str, dict],
    collision_threshold: int = 3,
) -> None:
    """Print a difficulty-group comparison table to the terminal.

    Parameters
    ----------
    groups_agg : dict
        Maps group label -> aggregated metrics dict.
    collision_threshold : int
        The collision threshold used (for display only).
    """
    key_metrics = [
        "success_rate", "oracle_success_rate",
        "sr_new",
        "spl_new",
        "tcr_new",
        "dcr_new",
        "avg_steps", "avg_travel_length", "avg_shortest_path_length",
        "avg_final_distance",
    ]

    labels = [k for k in groups_agg if groups_agg[k].get("count", 0) > 0]
    if not labels:
        print("No valid data to display")
        return

    col_width = 14
    header = (
        f"{'metric':<36}"
        + "".join(f"{lb:>{col_width}}" for lb in labels)
        + f"{'overall':>{col_width}}"
    )
    print(f"\n{'=' * len(header)}")
    print(f"Difficulty group comparison  collision_threshold={collision_threshold}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    all_counts = [groups_agg[lb].get("count", 0) for lb in labels]
    total_count = sum(all_counts)

    for metric in key_metrics:
        row = f"{metric:<36}"
        weighted_sum = 0.0
        for lb in labels:
            v = groups_agg[lb].get(metric, 0.0)
            row += f"{v:>{col_width}.4f}"
            weighted_sum += v * groups_agg[lb].get("count", 0)
        overall = weighted_sum / total_count if total_count > 0 else 0.0
        row += f"{overall:>{col_width}.4f}"
        print(row)

    row_count = f"{'count':<36}"
    for lb in labels:
        row_count += f"{groups_agg[lb].get('count', 0):>{col_width}d}"
    row_count += f"{total_count:>{col_width}d}"
    print("-" * len(header))
    print(row_count)
    print("=" * len(header))


# ============================================================================
# Public API -- full pipeline
# ============================================================================

def analyze_and_report(
    result_dir: str,
    mode: str = "outdoor",
    collision_threshold: Optional[int] = None,
    min_distance: float = 5.0,
    max_distance: float = 50.0,
    exclude_scenes: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    per_scene: bool = False,
) -> dict:
    """Run the full metrics analysis pipeline.

    1. Load all ``result.json`` files from *result_dir*.
    2. Exclude specified scenes.
    3. Classify by difficulty.
    4. Compute and aggregate metrics per group and overall.
    5. Print summary tables and save JSON output.

    Parameters
    ----------
    result_dir : str
        Root evaluation output directory.
    mode : str
        ``"outdoor"`` or ``"indoor"``.
    collision_threshold : int, optional
        Collision-count threshold for SR_NEW.  Auto-selected by *mode*
        if omitted (outdoor=3, indoor=1).
    min_distance, max_distance : float
        Distance range for outdoor difficulty classification.
    exclude_scenes : list of str, optional
        Scene IDs to exclude.  Defaults to ``OUTDOOR_EXCLUDED_SCENES``
        for outdoor mode, empty for indoor.
    output_path : str, optional
        Path for JSON output.  Defaults to
        ``{result_dir}/eval_summary.json``.
    per_scene : bool
        If True, print per-scene breakdown.

    Returns
    -------
    dict
        Full analysis payload (also saved as JSON).
    """
    from ..point_goal.evaluator import POINT_GOAL_PROTOCOL

    if collision_threshold is None:
        protocol = POINT_GOAL_PROTOCOL.get(mode, {})
        collision_threshold = protocol.get("collision_threshold", 3)

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
            r for r in results if r.get("episode_id", "") not in exclude_scenes
        ]
        print(
            f"Excluded scenes {exclude_scenes}: "
            f"filtered {before - len(results)} tasks, {len(results)} remaining"
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
        metrics_list = [extract_metrics(r, collision_threshold) for r in group_results]
        groups_agg[label] = aggregate(metrics_list)

    # Overall
    all_metrics = [extract_metrics(r, collision_threshold) for r in results]
    overall_agg = aggregate(all_metrics)

    # Print tables
    print_summary_table(groups_agg, collision_threshold)

    # Per-group detailed print
    print(f"\n{'=' * 60}")
    print("Per-group detailed metrics")
    print(f"{'=' * 60}")
    for label, agg in groups_agg.items():
        _print_group_metrics(label, agg)
    _print_group_metrics("overall", overall_agg)

    # Per-scene breakdown
    if per_scene:
        _print_per_scene_metrics(results, collision_threshold)

    # Status distribution across all tasks
    status_distribution: Dict[str, int] = {}
    for r in results:
        s = r.get("status", "unknown")
        status_distribution[s] = status_distribution.get(s, 0) + 1

    # Save unified summary JSON
    out_path = output_path or os.path.join(result_dir, "eval_summary.json")
    payload: Dict[str, Any] = {
        "task_type": "point_goal",
        "result_dir": os.path.abspath(result_dir),
        "mode": mode,
        "collision_threshold": collision_threshold,
        "total_count": len(results),
        "status_distribution": status_distribution,
    }
    if mode == "outdoor":
        payload["distance_config"] = {
            "min_distance": min_distance,
            "max_distance": max_distance,
            "step": (max_distance - min_distance) / 3.0,
        }
    payload["overall"] = overall_agg
    payload["groups"] = {label: agg for label, agg in groups_agg.items()}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved eval summary JSON: {out_path}")

    return payload


# ============================================================================
# Internal printing helpers
# ============================================================================

def _print_group_metrics(group_name: str, agg: dict) -> None:
    if agg.get("count", 0) == 0:
        print(f"\n  [{group_name}] no data")
        return
    print(f"\n  [{group_name}] ({agg['count']} tasks)")
    print(f"  {'metric':<36}{'value':>12}")
    print(f"  {'-' * 50}")
    for k, v in agg.items():
        if k == "count":
            continue
        if isinstance(v, (int, float)):
            print(f"  {k:<36}{v:>12.4f}")


def _print_per_scene_metrics(results: List[dict], collision_threshold: int) -> None:
    from collections import defaultdict

    scene_groups = defaultdict(list)
    for r in results:
        scene_groups[r.get("episode_id", "unknown")].append(r)

    print(f"\n{'=' * 80}")
    print("Per-scene metrics")
    print(f"{'=' * 80}")

    col_w = 10
    header = (
        f"{'scene':<20}{'count':>{col_w}}{'succ_rate':>{col_w}}"
        f"{'sr_new':>{col_w}}{'spl_new':>{col_w}}"
        f"{'tcr_new':>{col_w}}{'dcr_new':>{col_w}}{'avg_sp':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for scene_name in sorted(scene_groups):
        mlist = [extract_metrics(r, collision_threshold) for r in scene_groups[scene_name]]
        agg = aggregate(mlist)
        n = agg.get("count", 0)
        print(
            f"{scene_name:<20}{n:>{col_w}d}"
            f"{agg.get('success_rate', 0):>{col_w}.4f}"
            f"{agg.get('sr_new', 0):>{col_w}.4f}"
            f"{agg.get('spl_new', 0):>{col_w}.4f}"
            f"{agg.get('tcr_new', 0):>{col_w}.4f}"
            f"{agg.get('dcr_new', 0):>{col_w}.4f}"
            f"{agg.get('avg_shortest_path_length', 0):>{col_w}.1f}"
        )


def _cli_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Aggregate point-goal evaluation metrics. "
                    "Can be run at any time — even while evaluation is still in progress — "
                    "to get interim results from completed tasks.",
    )
    parser.add_argument("--result-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--mode", default="outdoor", choices=["outdoor", "indoor"])
    parser.add_argument("--collision-threshold", type=int, default=None,
                        help="Auto-selected by --mode if omitted (outdoor=3, indoor=1)")
    parser.add_argument("--min-distance", type=float, default=5.0)
    parser.add_argument("--max-distance", type=float, default=50.0)
    parser.add_argument("--exclude-scenes", nargs="*", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--per-scene", action="store_true")
    args = parser.parse_args(argv)
    analyze_and_report(
        result_dir=args.result_dir,
        mode=args.mode,
        collision_threshold=args.collision_threshold,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        exclude_scenes=args.exclude_scenes,
        output_path=args.output_path,
        per_scene=args.per_scene,
    )


if __name__ == "__main__":
    _cli_main()
