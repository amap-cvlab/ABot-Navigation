"""Command-line entry point for POI-goal evaluation.

Extends the point-goal runner with POI-specific parameters:

* ``--collision-mode`` (off/soft/hard, default hard)
* ``--robot-radius`` (metres, default 0.0)
* ``--arrive-threshold`` (default 2.0m)
* ``--occ-obstacle-polarity`` (dark/light)
* ``--occ-dark-threshold`` (int)
* ``--per-poi`` flag for per-POI metrics breakdown

Usage examples::

    # Basic run with a custom POI-goal agent
    python -m abotn_evaluator.poi_goal.runner \\
        --data-dir /path/to/scenes \\
        --render-url http://localhost:7001/render_gs \\
        --output-dir ./eval_output \\
        --agent-module my_agents.poi_nav:MyPoiAgent

    # With collision detection and custom threshold
    python -m abotn_evaluator.poi_goal.runner \\
        --data-dir /path/to/scenes \\
        --render-url http://localhost:7001/render_gs \\
        --output-dir ./eval_output \\
        --agent-module my_agents.poi_nav:MyPoiAgent \\
        --collision-mode hard \\
        --robot-radius 0.15 \\
        --arrive-threshold 2.0 \\
        --max-steps 100

    # With metrics analysis
    python -m abotn_evaluator.poi_goal.runner \\
        --data-dir /path/to/scenes \\
        --render-url http://localhost:7001/render_gs \\
        --output-dir ./eval_output \\
        --agent-module my_agents.poi_nav:MyPoiAgent \\
        --mode indoor \\
        --per-poi
"""

import argparse
import os
import sys
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="POI-goal navigation evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Data / output ---
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory containing per-scene subdirectories",
    )
    parser.add_argument(
        "--map-dir", type=str, default=None,
        help="Separate directory for map data (occ_map, height_map, etc.)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_output",
        help="Directory for evaluation outputs (default: ./eval_output)",
    )

    # --- Renderer ---
    parser.add_argument(
        "--render-url", type=str, default="http://127.0.0.1:7001/render_gs",
        help="URL of the Gaussian splatting render service",
    )

    # --- Agent ---
    parser.add_argument(
        "--agent-module", type=str, default=None,
        help="Agent class path as 'package.module:ClassName'",
    )
    parser.add_argument(
        "--agent-config", type=str, default=None,
        help="Optional YAML config file for the agent",
    )

    # --- Evaluator ---
    parser.add_argument(
        "--evaluator-module", type=str, default=None,
        help=(
            "Custom evaluator class path as 'package.module:ClassName'. "
            "Must accept (scene, renderer, config, output_dir) in constructor."
        ),
    )

    # --- Resume ---
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help=(
            "Directory of a previous run to resume. When omitted, a new "
            "timestamped subdirectory is created under --output-dir."
        ),
    )

    # --- POI-goal specific parameters ---
    parser.add_argument(
        "--arrive-threshold", type=float, default=2.0,
        help=(
            "Goal arrival distance threshold in metres (default: 2.0). "
            "Larger than point-goal because POI targets are typically at "
            "building facades."
        ),
    )
    parser.add_argument(
        "--collision-mode", type=str, default="hard",
        choices=["off", "soft", "hard"],
        help=(
            "Collision handling mode (default: hard). "
            "'off' = no collision detection; "
            "'soft' = count collisions only, no effect on success; "
            "'hard' = terminate on first non-exempt collision."
        ),
    )
    parser.add_argument(
        "--robot-radius", type=float, default=0.0,
        help=(
            "Robot radius in metres for circular footprint collision "
            "detection.  0.0 means only the centre pixel is checked. "
            "(default: 0.0)"
        ),
    )
    parser.add_argument(
        "--occ-obstacle-polarity", type=str, default="dark",
        choices=["dark", "light"],
        help=(
            "How to interpret occ_map pixel values: "
            "'dark' = dark pixels are obstacles (default); "
            "'light' = light pixels are obstacles."
        ),
    )
    parser.add_argument(
        "--occ-dark-threshold", type=int, default=64,
        help="Greyscale threshold for obstacle detection (default: 64)",
    )

    # --- Eval parameters ---
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum steps per task (default: 100)",
    )
    parser.add_argument(
        "--collision-threshold", type=int, default=3,
        help=(
            "Collision count threshold for legacy SR_NEW metrics "
            "(default: 3)"
        ),
    )

    # --- Camera ---
    parser.add_argument(
        "--camera-width", type=int, default=720,
        help="Camera image width (default: 720)",
    )
    parser.add_argument(
        "--camera-height", type=int, default=640,
        help="Camera image height (default: 640)",
    )
    parser.add_argument(
        "--camera-fx", type=float, default=252.075,
        help="Camera focal length x (default: 252.075)",
    )
    parser.add_argument(
        "--camera-fy", type=float, default=252.075,
        help="Camera focal length y (default: 252.075)",
    )
    parser.add_argument(
        "--extrinsic-height", type=float, default=0.65,
        help="Camera height above ground in metres (default: 0.65)",
    )

    # --- Observation options ---
    parser.add_argument(
        "--provide-history", action="store_true",
        help="Include history images and poses in observations",
    )
    parser.add_argument(
        "--provide-occ-map", action="store_true",
        help="Include occupancy map in observations",
    )
    parser.add_argument(
        "--provide-height-map", action="store_true",
        help="Include height map in observations",
    )
    parser.add_argument(
        "--max-history-frames", type=int, default=20,
        help="Max history frames in ShortMemory (default: 20)",
    )
    parser.add_argument(
        "--history-resize-ratio", type=float, default=0.25,
        help="Resize ratio for history frame images (default: 0.25)",
    )

    # --- Output options ---
    parser.add_argument(
        "--save-render-images", action="store_true", default=True,
        help="Save rendered images to disk (default: True)",
    )
    parser.add_argument(
        "--no-save-render-images", action="store_false",
        dest="save_render_images",
        help="Do not save rendered images to disk",
    )
    parser.add_argument(
        "--enable-visualization", action="store_true", default=False,
        help="Enable visualization of agent's extra field (e.g. affordance pixel overlay)",
    )

    # --- Occ-map dilation ---
    parser.add_argument(
        "--occ-dilation-meters", type=float, default=0.5,
        help="Dilate free space by this many metres (default: 0.5)",
    )

    # --- Metrics analysis ---
    parser.add_argument(
        "--mode", type=str, default="indoor",
        choices=["outdoor", "indoor"],
        help="Difficulty classification mode (default: indoor)",
    )
    parser.add_argument(
        "--min-distance", type=float, default=5.0,
        help="Outdoor: minimum path length for bucketing (default: 5.0)",
    )
    parser.add_argument(
        "--max-distance", type=float, default=50.0,
        help="Outdoor: maximum path length for bucketing (default: 50.0)",
    )
    parser.add_argument(
        "--exclude-scenes", type=str, nargs="*", default=None,
        help="Scene IDs to exclude from metrics",
    )
    parser.add_argument(
        "--per-scene", action="store_true",
        help="Print per-scene metrics breakdown",
    )
    parser.add_argument(
        "--per-poi", action="store_true", default=True,
        help="Print per-POI metrics breakdown (default: True)",
    )
    parser.add_argument(
        "--no-per-poi", action="store_false", dest="per_poi",
        help="Do not print per-POI metrics breakdown",
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip post-evaluation metrics analysis",
    )

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Load agent ---
    if args.agent_module:
        from abotn_evaluator.agent_loader import load_class
        agent_cls = load_class(args.agent_module)
        if args.agent_config:
            from abotn_evaluator.config import load_agent_config
            agent_config = load_agent_config(args.agent_config)
            agent_config["arrive_threshold"] = args.arrive_threshold
            agent = agent_cls(**agent_config)
        else:
            agent = agent_cls()
    else:
        print("Error: --agent-module must be specified")
        sys.exit(1)

    # --- Build camera config ---
    from abotn_evaluator.render_client import CameraConfig

    camera_config = CameraConfig(
        width=args.camera_width,
        height=args.camera_height,
        fx=args.camera_fx,
        fy=args.camera_fy,
        extrinsic_height=args.extrinsic_height,
    )

    # --- Build eval config ---
    from .evaluator import PoiGoalEvalConfig

    eval_config = PoiGoalEvalConfig(
        render_url=args.render_url,
        max_steps=args.max_steps,
        arrive_threshold=args.arrive_threshold,
        camera_config=camera_config,
        collision_threshold=args.collision_threshold,
        provide_history=args.provide_history,
        provide_occ_map=args.provide_occ_map,
        provide_height_map=args.provide_height_map,
        max_history_frames=args.max_history_frames,
        history_resize_ratio=args.history_resize_ratio,
        save_render_images=args.save_render_images,
        occ_dilation_meters=args.occ_dilation_meters,
        enable_visualization=args.enable_visualization,
        # POI-specific config
        collision_mode=args.collision_mode,
        robot_radius=args.robot_radius,
        occ_obstacle_polarity=args.occ_obstacle_polarity,
        occ_dark_threshold=args.occ_dark_threshold,
    )

    effective_mode = eval_config.resolve_collision_mode()
    if effective_mode != "off":
        print(
            f"[PoiGoal] collision check ENABLED "
            f"(mode={effective_mode}): "
            f"robot_radius={args.robot_radius}m, "
            f"polarity={args.occ_obstacle_polarity}, "
            f"thr={args.occ_dark_threshold}"
        )
    else:
        print("[PoiGoal] collision check DISABLED")

    # --- Build scene ---
    from abotn_evaluator.scene import GaussianScene

    scene = GaussianScene(
        local_data_path=args.data_dir,
        local_map_path=args.map_dir,
    )
    print(f"Scene summary: {scene.summary()}")

    # --- Build renderer ---
    from abotn_evaluator.render_client import GaussianRenderer

    renderer = GaussianRenderer(
        render_url=args.render_url,
        camera_config=camera_config,
    )

    # --- Resolve output directory ---
    if args.resume_dir:
        effective_output_dir = args.resume_dir
        print(f"[resume] Resuming from: {effective_output_dir}")
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_output_dir = os.path.join(args.output_dir, timestamp)
        print(f"[new run] Output directory: {effective_output_dir}")
    os.makedirs(effective_output_dir, exist_ok=True)

    # --- Build evaluator ---
    if args.evaluator_module:
        evaluator_cls = load_class(args.evaluator_module)
        evaluator = evaluator_cls(
            scene=scene,
            renderer=renderer,
            config=eval_config,
            output_dir=effective_output_dir,
        )
    else:
        from .evaluator import PoiGoalEvaluator
        evaluator = PoiGoalEvaluator(
            scene=scene,
            renderer=renderer,
            config=eval_config,
            output_dir=effective_output_dir,
        )

    # --- Run evaluation ---
    results = evaluator.evaluate(agent, resume_dir=args.resume_dir)
    print(f"\nEvaluation complete. {len(results)} tasks evaluated.")

    # --- Metrics analysis ---
    if not args.skip_metrics:
        from .metrics import analyze_and_report

        report = analyze_and_report(
            result_dir=effective_output_dir,
            mode=args.mode,
            collision_threshold=args.collision_threshold,
            arrive_threshold=args.arrive_threshold,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            exclude_scenes=args.exclude_scenes,
            per_scene=args.per_scene,
            per_poi=args.per_poi,
        )
        print(
            f"\nMetrics analysis complete. "
            f"Overall count: {report.get('total_count', 0)}"
        )


if __name__ == "__main__":
    main()
