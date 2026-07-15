"""
Short-term memory buffer for multi-view RGB images and agent poses.

Manages a sliding window of observation frames, where each frame consists of
multiple camera views (left, front, right) and an associated pose. Historical
frames are downscaled to save memory while current-frame images remain at
full resolution.
"""

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default constants (overridable via constructor)
DEFAULT_MAX_HISTORY_FRAMES = 20
DEFAULT_NUM_CURRENT_VIEWS = 3
DEFAULT_INPUT_IMG_SIZE = (476, 420)
DEFAULT_HISTORY_RESIZE_RATIO = 0.25

# The input view order is [left, right, front].
# We reorder to [left, front, right] for consistency.
_VIEW_REORDER_INDICES = [0, 2, 1]


class ShortMemory:
    """Sliding-window memory buffer for multi-view RGB observations and poses.

    Each *frame* contains ``num_current_views`` images (default 3: left, front,
    right) plus one pose.  The buffer keeps at most ``max_history_frames``
    historical frames.  When the window is full, either the oldest frame is
    dropped (FIFO mode, when ``have_memory=True``) or the frame pair with the
    smallest temporal gap is merged (adaptive mode).

    Historical images are downscaled by ``resize_ratio`` to reduce memory usage
    while current-frame images are kept at their original resolution.

    Parameters
    ----------
    max_history_frames : int
        Maximum number of *history* frames to retain (excluding the current
        frame).
    num_current_views : int
        Number of camera views per frame (e.g. 3 for left/front/right).
    input_img_size : tuple of (int, int)
        (width, height) of the input images.  Used as the reference size when
        downscaling history frames.
    resize_ratio : float
        Scale factor applied to history frame images (e.g. 0.25 means images
        are shrunk to 25 % of their original dimensions).
    reorder_views : bool
        If True (default), reorder incoming views from [left, right, front] to
        [left, front, right].
    resize_on_add : bool
        If True (default), historical frames are downscaled when a new frame
        is added.  Set to False when external code handles resizing (e.g.
        instruction-goal agents that use ``build_qwen_messages()``).
    """

    def __init__(
        self,
        max_history_frames: int = DEFAULT_MAX_HISTORY_FRAMES,
        num_current_views: int = DEFAULT_NUM_CURRENT_VIEWS,
        input_img_size: Tuple[int, int] = DEFAULT_INPUT_IMG_SIZE,
        resize_ratio: float = DEFAULT_HISTORY_RESIZE_RATIO,
        reorder_views: bool = True,
        resize_on_add: bool = True,
    ) -> None:
        self.max_history_frames = max_history_frames
        self.num_current_views = num_current_views
        self.input_img_size = input_img_size
        self.resize_ratio = resize_ratio
        self.reorder_views = reorder_views
        self.resize_on_add = resize_on_add

        # Internal storage
        self._rgb_list: List[Image.Image] = []
        self._pose_list: List[np.ndarray] = []
        self._depth_raw_list: List[np.ndarray] = []
        self._image_indices: List[int] = []
        self._total_frame_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total number of frames that have been added (including evicted)."""
        return self._total_frame_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored images, poses, and counters."""
        self._rgb_list = []
        self._pose_list = []
        self._depth_raw_list = []
        self._image_indices = []
        self._total_frame_count = 0

    def reset_history(self) -> None:
        """Reset history, keeping only the current frame's views."""
        n = self.num_current_views
        self._rgb_list = self._rgb_list[-n:]
        self._pose_list = self._pose_list[-n:]
        self._depth_raw_list = self._depth_raw_list[-n:] if self._depth_raw_list else []
        self._image_indices = self._image_indices[-n:]

    def add_frame(
        self,
        rgbs: List,
        pose: np.ndarray,
        have_memory: bool = False,
        num_current_image: Optional[int] = None,
    ) -> int:
        """Append a new multi-view observation frame to the buffer.

        Parameters
        ----------
        rgbs : list of ndarray or PIL.Image
            List of camera-view images for the current timestep.  Expected
            length equals ``num_current_views``.  Each element is either a
            NumPy HWC uint8 array or a PIL Image.
        pose : np.ndarray
            Camera-to-world transform (or any pose representation) associated
            with this frame.
        have_memory : bool
            Eviction strategy when the window is full.
            * True  -- FIFO: drop the oldest frame.
            * False -- Adaptive: drop the frame whose temporal gap to its
              neighbour is smallest, preserving temporal spread.
        num_current_image : int, optional
            Override the number of current views for this call.  When *None*
            the constructor value ``num_current_views`` is used.  Provided
            for backward compatibility with model-side callers.

        Returns
        -------
        int
            The zero-based frame index assigned to this observation.
        """
        t0 = time.time()

        # Allow callers to override num_current_views per-call
        n = num_current_image if num_current_image is not None else self.num_current_views

        # --- Convert / normalise images --------------------------------
        processed: List[Image.Image] = []
        for rgb in rgbs:
            if isinstance(rgb, np.ndarray):
                rgb = Image.fromarray(rgb)
            processed.append(rgb)

        # Reorder views: input [left, right, front] -> [left, front, right]
        if self.reorder_views and len(processed) >= 3:
            processed = [processed[i] for i in _VIEW_REORDER_INDICES]

        # --- Evict side views of the previous "current" frame ----------
        #
        # At any point the list stores:
        #   [history_front, ..., history_front, cur_left, cur_front, cur_right]
        # Before appending a new frame we collapse the old current frame to
        # just its front view so that history only keeps front views.
        if len(self._rgb_list) >= n:
            # Remove left and right views of the outgoing current frame
            # (indices -1 and -2 relative to end, i.e. right then left).
            for offset in [-1, -2]:
                self._rgb_list.pop(offset)
                self._pose_list.pop(offset)
                self._image_indices.pop(offset)

        # --- Append new frame ------------------------------------------
        current_frame_index = self._total_frame_count
        self._rgb_list.extend(processed)
        self._pose_list.extend([pose] * len(processed))
        self._image_indices.extend([current_frame_index] * len(processed))
        self._total_frame_count += 1

        # --- Downscale the frame that just became "history" -------------
        if self.resize_on_add and len(self._rgb_list) > n:
            hist_idx = -(1 + n)
            w, h = self._rgb_list[hist_idx].size
            new_size = (
                int(w * self.resize_ratio),
                int(h * self.resize_ratio),
            )
            self._rgb_list[hist_idx] = self._rgb_list[hist_idx].resize(new_size)

        # --- Evict if over capacity ------------------------------------
        if len(self._rgb_list) > self.max_history_frames + n:
            if have_memory:
                # FIFO: drop the oldest single image entry
                self._rgb_list.pop(0)
                self._pose_list.pop(0)
                self._image_indices.pop(0)
            else:
                # Adaptive: remove the history frame whose temporal gap to its
                # predecessor is smallest, preserving temporal coverage.
                history_indices = self._image_indices[: -n]
                min_gap_pos = int(np.argmin(np.diff(history_indices)))
                self._rgb_list.pop(min_gap_pos + 1)
                self._pose_list.pop(min_gap_pos + 1)
                self._image_indices.pop(min_gap_pos + 1)

        elapsed = time.time() - t0
        logger.debug(
            "add_frame #%d  indices=%s  (%.4fs)",
            current_frame_index,
            self._image_indices,
            elapsed,
        )
        return current_frame_index

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_all_images(self) -> List[Image.Image]:
        """Return the full list of stored images (history + current views)."""
        return list(self._rgb_list)

    def get_current_images(self) -> List[Image.Image]:
        """Return only the current frame's multi-view images."""
        n = self.num_current_views
        if len(self._rgb_list) < n:
            return list(self._rgb_list)
        return list(self._rgb_list[-n:])

    def get_history_images(self) -> List[Image.Image]:
        """Return only the historical (downscaled) images, excluding current."""
        n = self.num_current_views
        if len(self._rgb_list) <= n:
            return []
        return list(self._rgb_list[: -n])

    def get_history_poses(self) -> List[np.ndarray]:
        """Return poses corresponding to the historical frames."""
        n = self.num_current_views
        if len(self._pose_list) <= n:
            return []
        return list(self._pose_list[: -n])

    def get_all_poses(self) -> List[np.ndarray]:
        """Return the full list of stored poses."""
        return list(self._pose_list)

    def get_last_pose(self) -> Optional[np.ndarray]:
        """Return the most recently added pose, or None if empty."""
        if self._pose_list:
            return self._pose_list[-1]
        return None

    def get_image_indices(self) -> List[int]:
        """Return the frame-index label for every stored image entry."""
        return list(self._image_indices)

    def get_depth_raw_list(self) -> List[np.ndarray]:
        """Return the full list of stored raw depth arrays."""
        return list(self._depth_raw_list)

    # ------------------------------------------------------------------
    # Backward-compatible aliases (model-side API)
    # ------------------------------------------------------------------

    def get_rgb_list(self) -> List[Image.Image]:
        """Alias for :meth:`get_all_images` (model-side compatibility)."""
        return self.get_all_images()

    def get_pose_list(self) -> List[np.ndarray]:
        """Alias for :meth:`get_all_poses` (model-side compatibility)."""
        return self.get_all_poses()

    def get_total_frame_count(self) -> int:
        """Alias for :attr:`frame_count` (model-side compatibility)."""
        return self.frame_count

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of individual image entries currently stored."""
        return len(self._rgb_list)

    def __repr__(self) -> str:
        return (
            f"ShortMemory(frames_added={self._total_frame_count}, "
            f"images_stored={len(self._rgb_list)}, "
            f"max_history={self.max_history_frames})"
        )
