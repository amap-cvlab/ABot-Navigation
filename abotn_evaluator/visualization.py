"""
Visualization helpers for navigation experiments.

Provides:
- ``process_extra_visualization`` -- generic dispatcher that reads
  ``WaypointPrediction.extra`` and calls registered handlers.
- ``plot_trajectory_on_occ_map`` -- trajectory overlay on occupancy maps.
- ``render_multiview_strip`` -- horizontal image strip rendering.

To add a new visualization type, register a handler in ``_HANDLERS``.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


# ============================================================================
# Extra-field visualization dispatcher
# ============================================================================

def process_extra_visualization(
    extra: Dict[str, Any],
    images: List,
    save_dir: str,
    step_id: int,
) -> None:
    """Process agent's extra dict and produce visualization outputs.

    Called by the evaluator when ``enable_visualization`` is True and
    ``prediction.extra`` is non-empty.  Dispatches to registered handlers
    based on the keys present in *extra*.

    Parameters
    ----------
    extra : dict
        The ``WaypointPrediction.extra`` dict from the agent.
    images : list of PIL.Image
        Current multi-view rendered images ``[left, front, right]``.
    save_dir : str
        Directory where visualization images should be saved.
    step_id : int
        Current step identifier (used for file naming).
    """
    for key, handler in _HANDLERS.items():
        if key in extra:
            try:
                handler(extra[key], images, save_dir, step_id)
            except Exception as exc:
                logger.warning("Visualization handler %r failed: %s", key, exc)


# ============================================================================
# Built-in handler: affordance_pixel
# ============================================================================

def _handle_affordance_pixel(
    data: Any,
    images: List,
    save_dir: str,
    step_id: int,
) -> None:
    """Draw affordance pixel markers on rendered images and save.

    Parameters
    ----------
    data : dict
        Pixel coordinates per view.  Supports formats:
        - ``{"front": [u, v], "left": [u, v], ...}``
        - ``{"Affordance Pixel": {"front": [u, v], ...}}``
    images : list of PIL.Image
        Current rendered images ``[left, front, right]``.
    save_dir : str
        Output directory.
    step_id : int
        Step number for file naming.
    """
    import os
    view_names = ("left", "front", "right")
    for idx, vname in enumerate(view_names):
        if idx >= len(images):
            continue
        px = _extract_pixel_coord(data, vname)
        if px is not None:
            img = images[idx].copy()
            _draw_pixel_marker(img, px)
            img.save(os.path.join(save_dir, f"{step_id}_{vname}.jpg"))


def _extract_pixel_coord(
    pixel_data: Any, view_name: str
) -> Optional[Tuple[float, float]]:
    """Extract pixel coordinates for a given view from various formats.

    Supports wrapped formats (``{"Affordance Pixel": {"left": ...}}``)
    and flat formats (``{"left": ..., "front": ...}``), with
    ``left/front/right`` or ``leftside/frontside/rightside`` keys in
    any casing.
    """
    aliases = {
        "left": ["left", "Left", "LEFT", "leftside", "Leftside", "LEFTSIDE"],
        "right": ["right", "Right", "RIGHT", "rightside", "Rightside", "RIGHTSIDE"],
        "front": ["front", "Front", "FRONT", "frontside", "Frontside", "FRONTSIDE"],
    }.get(view_name, [view_name, view_name.capitalize(), view_name.lower()])

    sections = []
    if isinstance(pixel_data, dict):
        for key in ("Affordance Pixel", "Target Pixel"):
            if key in pixel_data:
                sections.append(pixel_data[key])
        if any(
            k in pixel_data
            for k in ("left", "front", "right", "leftside", "frontside", "rightside")
        ):
            sections.append(pixel_data)

    for section in sections:
        if isinstance(section, str):
            try:
                section = json.loads(section)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(section, dict):
            continue
        for k in aliases:
            if k in section:
                val = section[k]
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    try:
                        return (float(val[0]), float(val[1]))
                    except (TypeError, ValueError):
                        return None
    return None


def _draw_pixel_marker(
    img: Image.Image,
    pixel: Tuple[float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 6,
) -> None:
    """Draw a colored marker on *img* at *pixel*.

    Handles three coordinate conventions:
    - normalized [0, 1]
    - Qwen-style [0, 1000]
    - absolute pixel coordinates
    """
    u, v = float(pixel[0]), float(pixel[1])
    w, h = img.size
    max_val = max(abs(u), abs(v))
    if max_val <= 1.0:
        u_px, v_px = int(u * w), int(v * h)
    elif max_val <= 1000.0:
        u_px, v_px = int(u / 1000.0 * w), int(v / 1000.0 * h)
    else:
        u_px, v_px = int(round(u)), int(round(v))

    u_px = max(radius, min(w - radius - 1, u_px))
    v_px = max(radius, min(h - radius - 1, v_px))

    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [u_px - radius, v_px - radius, u_px + radius, v_px + radius],
        fill=color,
    )

    label = "afford point"
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_offset = radius + 4
    draw.text(
        (u_px, v_px - text_offset),
        label,
        fill=color,
        font=font,
        anchor="mb" if font is not None else None,
    )


# Handler registry -- add new visualization types here.
_HANDLERS = {
    "affordance_pixel": _handle_affordance_pixel,
}


def plot_trajectory_on_occ_map(
    occ_map: np.ndarray,
    trajectory: Union[np.ndarray, List[Tuple[float, float]]],
    goal: Optional[Tuple[float, float]] = None,
    start: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (8, 8),
    trajectory_color: str = "blue",
    goal_color: str = "red",
    start_color: str = "green",
    linewidth: float = 2.0,
    point_size: float = 80.0,
    title: str = "Trajectory on Occupancy Map",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[np.ndarray]:
    """Plot an agent trajectory overlaid on a 2-D occupancy map.

    Parameters
    ----------
    occ_map : np.ndarray
        2-D occupancy grid (HxW).  Values are interpreted as grayscale
        intensities; 0 = free, 1 (or 255) = occupied.
    trajectory : array-like of shape (N, 2)
        Sequence of (x, y) positions in map coordinates.
    goal : tuple of (float, float), optional
        Goal position to mark on the map.
    start : tuple of (float, float), optional
        Start position to mark on the map.  If not given, the first
        trajectory point is used.
    figsize : tuple of (int, int)
        Figure size in inches.
    trajectory_color : str
        Color for the trajectory line.
    goal_color : str
        Color for the goal marker.
    start_color : str
        Color for the start marker.
    linewidth : float
        Width of the trajectory line.
    point_size : float
        Marker size for start/goal points.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this file path.
    show : bool
        If True, display the plot interactively (calls ``plt.show()``).

    Returns
    -------
    np.ndarray or None
        The rendered figure as an RGB numpy array (HxWx3) if matplotlib is
        available, otherwise None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed; skipping trajectory visualization."
        )
        return None

    traj = np.asarray(trajectory)
    if traj.ndim != 2 or traj.shape[1] < 2:
        logger.warning(
            "trajectory must have shape (N, 2); got %s. Skipping plot.",
            traj.shape,
        )
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Render occupancy map as grayscale background
    ax.imshow(occ_map, cmap="gray", origin="upper")

    # Draw trajectory
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=trajectory_color,
        linewidth=linewidth,
        label="Trajectory",
        zorder=2,
    )

    # Mark start
    start_pt = start if start is not None else (traj[0, 0], traj[0, 1])
    ax.scatter(
        [start_pt[0]],
        [start_pt[1]],
        color=start_color,
        s=point_size,
        marker="o",
        label="Start",
        zorder=3,
    )

    # Mark goal
    if goal is not None:
        ax.scatter(
            [goal[0]],
            [goal[1]],
            color=goal_color,
            s=point_size,
            marker="*",
            label="Goal",
            zorder=3,
        )

    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rendered = np.asarray(buf)[:, :, :3].copy()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved trajectory plot to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)
    return rendered


def render_multiview_strip(
    images: list,
    titles: Optional[List[str]] = None,
    figsize_per_image: Tuple[int, int] = (4, 4),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[np.ndarray]:
    """Render a horizontal strip of images (e.g. left/front/right views).

    Parameters
    ----------
    images : list of np.ndarray or PIL.Image
        Images to display side by side.
    titles : list of str, optional
        Subplot titles.  Must match the length of *images* if provided.
    figsize_per_image : tuple of (int, int)
        Figure size contribution per image (width, height) in inches.
    save_path : str, optional
        If provided, save the figure to this file path.
    show : bool
        If True, display the plot interactively.

    Returns
    -------
    np.ndarray or None
        The rendered strip as an RGB numpy array, or None if matplotlib is
        unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed; skipping multiview strip rendering."
        )
        return None

    n = len(images)
    if n == 0:
        return None

    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_image[0] * n, figsize_per_image[1]),
    )
    if n == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        img_arr = np.asarray(img)
        ax.imshow(img_arr)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i])

    fig.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rendered = np.asarray(buf)[:, :, :3].copy()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved multiview strip to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)
    return rendered
