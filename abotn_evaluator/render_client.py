"""Render client for the 3DGS render server.

Sends render requests to a remote Gaussian splatting service and returns
RGB images. Combines camera configuration, coordinate transforms, and
HTTP request handling in a single module.

This is the CLIENT side. The SERVER is in ``render_server/``.
"""

import base64
import io
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation


# ============================================================================
# Camera Configuration
# ============================================================================

@dataclass
class CameraConfig:
    """Pinhole camera configuration.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length along the x-axis (pixels).
        fy: Focal length along the y-axis (pixels).
        cx: Principal point x coordinate (pixels).
        cy: Principal point y coordinate (pixels).
        extrinsic_height: Camera height offset relative to the ground plane
            (metres).  Positive means above the ground.
    """

    width: int = 720
    height: int = 640
    fx: float = 252.075
    fy: float = 252.075
    cx: float = 360.0
    cy: float = 320.0
    extrinsic_height: float = 0.65

    @property
    def intrinsics_colmap(self) -> str:
        """Return the intrinsics formatted as a COLMAP camera line.

        Format: ``<camera_id> PINHOLE <width> <height> <fx> <fy> <cx> <cy>``
        """
        return (
            f"1 PINHOLE {self.width} {self.height} "
            f"{self.fx} {self.fy} {self.cx} {self.cy}"
        )

    def __repr__(self) -> str:
        return (
            f"CameraConfig(width={self.width}, height={self.height}, "
            f"fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, "
            f"extrinsic_height={self.extrinsic_height})"
        )


# ============================================================================
# Coordinate Transforms
# ============================================================================

# From USD camera convention to World camera convention.
W_U_TRANSFORM = np.array(
    [[0, 0, -1, 0],
     [-1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]
)

# From World camera convention to USD camera convention.
U_W_TRANSFORM = np.array(
    [[0, -1, 0, 0],
     [0, 0, 1, 0],
     [-1, 0, 0, 0],
     [0, 0, 0, 1]]
)


def _radians_to_degrees(radians: float) -> float:
    """Convert an angle from radians to degrees."""
    return radians * (180.0 / math.pi)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a scalar-first quaternion [qw, qx, qy, qz]."""
    qw = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    if qw == 0.0:
        return np.array([0.0, 1.0, 0.0, 0.0])
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
    return np.array([qw, qx, qy, qz])


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion [qw, qx, qy, qz] to a 4x4 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x), 0],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2), 0],
        [0, 0, 0, 1],
    ])


def _translation_matrix(translation: np.ndarray) -> np.ndarray:
    """Create a 4x4 translation matrix from a 3-element vector."""
    mat = np.eye(4)
    mat[:3, 3] = translation
    return mat


def _rotation_matrix_to_euler(mat: np.ndarray) -> np.ndarray:
    """Extract Euler angles (degrees) from a 3x3 rotation matrix.

    Uses the XYZ convention. Returns ``[rx, ry, rz]`` in degrees.
    """
    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(mat[2, 1], mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0
    return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])


def _rot_matrices_to_quats(rotation_matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to scalar-first quaternions."""
    rot = Rotation.from_matrix(rotation_matrices)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    return result


def _quats_to_rot_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternions to rotation matrices."""
    if len(quaternions.shape) == 1:
        q = quaternions[[1, 2, 3, 0]]
    else:
        q = quaternions[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(q)
    return rot.as_matrix()


def _is_yaw_value_valid(yaw_degrees: float) -> bool:
    """Return True if *yaw_degrees* is close to a multiple of 90 degrees."""
    valid_values = [0.0, 90.0, -90.0, 180.0, -180.0, 270.0, -270.0, 360.0]
    return any(math.isclose(yaw_degrees, v, abs_tol=1e-3) for v in valid_values)


def _world_to_camera_orientation(camera_world_orientation_w: np.ndarray) -> np.ndarray:
    """Transform a world-convention quaternion to USD camera convention."""
    camera_world_orientation_w = np.asarray(camera_world_orientation_w, dtype=np.float32)
    world_w_cam_w_R = _quats_to_rot_matrices(camera_world_orientation_w)
    w_u_R = W_U_TRANSFORM[:3, :3].astype(np.float32)
    calc_w_to_usd_orientation = _rot_matrices_to_quats(np.matmul(world_w_cam_w_R, w_u_R))
    return calc_w_to_usd_orientation


def xyz_euler_trans_gs_colmap(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    degree_flag: bool = False,
) -> str:
    """Convert a world-frame pose to a COLMAP extrinsics string.

    The returned string has the format expected by the Gaussian splatting
    render service::

        <image_id> <qw> <qx> <qy> <qz> <tx> <ty> <tz> <camera_id> <image_name>

    Args:
        x, y, z: Camera position in world coordinates.
        roll, pitch, yaw: Euler angles.  In *radians* by default; set
            *degree_flag* to ``True`` to pass degrees instead.
        degree_flag: If ``True``, interpret roll/pitch/yaw as degrees.

    Returns:
        COLMAP-format extrinsics string.
    """
    if degree_flag:
        pitch_value = pitch
        roll_value = roll
        yaw_value = yaw
    else:
        pitch_value = _radians_to_degrees(pitch)
        roll_value = _radians_to_degrees(roll)
        yaw_value = _radians_to_degrees(yaw)

    gs_T = np.array([x, y, z])

    if _is_yaw_value_valid(yaw_value):
        euler_angles_w = np.array([roll_value, pitch_value, yaw_value + 0.001])
    else:
        euler_angles_w = np.array([roll_value, pitch_value, yaw_value])

    RR = Rotation.from_euler("xyz", euler_angles_w, degrees=True).as_matrix()
    quaternion_w = _rotation_matrix_to_quaternion(RR)
    quaternion_usd = _world_to_camera_orientation(quaternion_w)

    vec3d_pose = np.array([gs_T[0], gs_T[1], gs_T[2]])
    quat = np.array([
        float(quaternion_usd[0]),
        float(quaternion_usd[1]),
        float(quaternion_usd[2]),
        float(quaternion_usd[3]),
    ])

    rotation_mat = _quaternion_to_rotation_matrix(quat).T
    translation_mat = _translation_matrix(vec3d_pose).T

    camera_to_world_mat = rotation_mat @ translation_mat

    camera_to_object_pos = camera_to_world_mat[-1, :3]
    camera_to_object_mat_rot = camera_to_world_mat[:3, :3].T

    camera_to_object_rot = _rotation_matrix_to_euler(camera_to_object_mat_rot)

    pose_data = {
        "position": list(camera_to_object_pos),
        "rotation": list(np.deg2rad(camera_to_object_rot)),
    }

    position_t = (np.array(pose_data["position"]),)
    euler_angles = np.array(pose_data["rotation"])

    C2W = np.eye(4)
    C2W[:3, :3] = Rotation.from_euler("xyz", euler_angles).as_matrix()
    C2W[:3, 3] = position_t[0]
    W2C = np.linalg.inv(C2W)

    ISAAC_SIM_TO_GS_CONVENTION = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])

    W2C = ISAAC_SIM_TO_GS_CONVENTION @ W2C

    R = W2C[:3, :3]
    T = W2C[:3, 3]

    quaternion = _rotation_matrix_to_quaternion(R)

    camera_id = 1
    image_id = 1
    image_name = f"image_{1}.jpg"

    camera_image_pose_colmap = (
        f"{image_id} "
        + " ".join(map(str, quaternion))
        + " "
        + " ".join(map(str, T))
        + f" {camera_id} {image_name}"
    )

    return camera_image_pose_colmap


# ============================================================================
# Exceptions
# ============================================================================

class RenderFailureError(Exception):
    """Raised when the Gaussian splatting render service fails persistently."""
    pass


# ============================================================================
# Renderer
# ============================================================================

class GaussianRenderer:
    """Client for a remote Gaussian splatting render service.

    Args:
        render_url: Full URL of the render endpoint (e.g.
            ``"http://host:7001/render_gs"``).
        camera_config: Camera intrinsics/extrinsics configuration.  Defaults
            to a standard :class:`CameraConfig`.
        num_views: Number of views to render per pose.  Use ``3`` for
            left / right / front, or ``1`` for front only.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries on render failure.
        retry_backoff: Base sleep time (seconds) between retries; doubled
            after each attempt.
    """

    def __init__(
        self,
        render_url: str,
        camera_config: Optional[CameraConfig] = None,
        num_views: int = 3,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.render_url = render_url
        self.camera_config = camera_config or CameraConfig()
        self.num_views = num_views

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self.yaw_offsets = [90, -90, 0] if num_views == 3 else [0]
        self.view_names = (
            ["left", "right", "front"] if num_views == 3 else ["front"]
        )

    def render_at_pose(
        self,
        pose: np.ndarray,
        scene_id: str,
        save_dir: Optional[str] = None,
        image_id: int = 0,
        save_img_idx: Optional[List[int]] = None,
    ) -> List[Image.Image]:
        """Render images at a given 4x4 pose matrix.

        For multi-view setups (``num_views=3``), the renderer produces left,
        right, and front views by adding fixed yaw offsets.

        Args:
            pose: 4x4 camera-to-world transformation matrix.
            scene_id: Identifier of the scene to render.
            save_dir: If provided, rendered images are saved to this directory.
            image_id: Numeric prefix used when saving images.
            save_img_idx: If set, only images whose index appears in this list
                are saved to disk.

        Returns:
            List of rendered PIL images.
        """
        images: List[Image.Image] = []

        x, y, z = pose[:3, 3]
        roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler(
            "xyz", degrees=False
        )

        for idx, yaw_offset in enumerate(self.yaw_offsets):
            current_yaw = yaw + math.radians(yaw_offset)
            cam_extrinsics = xyz_euler_trans_gs_colmap(
                x, y, z, roll, pitch, current_yaw
            )
            img = self._render_request(cam_extrinsics, scene_id)
            if img is None:
                raise RenderFailureError(
                    f"render_at_pose got None from scene={scene_id}, "
                    f"yaw_offset={yaw_offset}"
                )
            images.append(img)

        if save_dir:
            for idx, img in enumerate(images):
                if save_img_idx is None or idx in save_img_idx:
                    img.save(
                        os.path.join(
                            save_dir, f"{image_id}_{self.view_names[idx]}.jpg"
                        )
                    )

        return images

    def _render_request(
        self,
        cam_extrinsics: str,
        scene_id: str,
        need_depth: bool = False,
    ) -> Union[
        Optional[Image.Image],
        Tuple[Optional[Image.Image], Optional[Image.Image], Optional[np.ndarray]],
        None,
    ]:
        """Send a single render request to the remote service."""
        intrinsics = self.camera_config.intrinsics_colmap

        payload = {
            "cam_extrinsics": cam_extrinsics,
            "cam_intrinsics": intrinsics,
            "scene_id": scene_id,
            "return_depth_img": need_depth,
            "return_depth_npz": need_depth,
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.render_url, json=payload, timeout=self.timeout
                )
                if response.status_code == 200:
                    if need_depth:
                        data = response.json()
                        rgb_bytes = base64.b64decode(data["image"])
                        render_img = Image.open(io.BytesIO(rgb_bytes))

                        depth_image = None
                        if "depth_img" in data:
                            depth_img_bytes = base64.b64decode(data["depth_img"])
                            depth_image = Image.open(io.BytesIO(depth_img_bytes))

                        depth_array = None
                        if "depth_npz" in data:
                            depth_npz_bytes = base64.b64decode(data["depth_npz"])
                            depth_array = np.load(io.BytesIO(depth_npz_bytes))["depth"]

                        return render_img, depth_image, depth_array
                    else:
                        image_bytes = io.BytesIO(response.content)
                        render_img = Image.open(image_bytes)
                        return render_img
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_error = str(e)
                print(f"Render request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

            if attempt < self.max_retries:
                sleep_time = self.retry_backoff * (2 ** attempt)
                print(f"Retrying render in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        print(
            f"All {self.max_retries + 1} render attempts failed for scene={scene_id}. "
            f"Last error: {last_error}"
        )
        return None
