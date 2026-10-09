"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    p = np.asarray(x[:3]).reshape(3, 1)
    
    # Đổi điểm từ vehicle frame sang sensor frame: p_s = R @ p + t
    R = sensor.veh_to_sens[:3, :3]
    t = sensor.veh_to_sens[:3, 3:]
    p_s = R @ p + t
    x_s, y_s, z_s = p_s.flatten()

    # Kiểm tra tọa độ hữu hạn
    if not np.all(np.isfinite(p_s)):
        return False

    # Camera yêu cầu độ sâu dương > 1e-6
    is_camera = hasattr(sensor, "f_i") or hasattr(sensor, "f_j")
    if is_camera and x_s <= 1e-6:
        return False

    # Góc ngang
    angle = np.arctan2(y_s, x_s)
    
    # Kiểm tra góc nằm trong sensor.fov [min_angle, max_angle]
    fov_min, fov_max = sensor.fov[0], sensor.fov[1]
    return bool(fov_min <= angle <= fov_max)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    p = np.asarray(x[:3]).reshape(3, 1)
    
    # Đổi điểm sang sensor frame: p_s = R @ p + t
    R = sensor.veh_to_sens[:3, :3]
    t = sensor.veh_to_sens[:3, 3:]
    p_s = R @ p + t
    x_s, y_s, z_s = p_s.flatten()

    # Kiểm tra tính hữu hạn và độ sâu trước khi chiếu
    if not np.all(np.isfinite(p_s)):
        raise ValueError(f"Non-finite camera sensor coordinates: p_s={p_s.tolist()}")
    if x_s <= 1e-6:
        raise ValueError(f"Camera depth x_s <= 1e-6: x_s={x_s}")

    # Mô hình camera pinhole
    u = sensor.c_i - sensor.f_i * (y_s / x_s)
    v = sensor.c_j - sensor.f_j * (z_s / x_s)

    return np.matrix([[u], [v]])


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    
    z_mat = np.matrix([[float(z[0])], [float(z[1])]])
    
    sigma_i = params.sigma_cam_i
    sigma_j = params.sigma_cam_j
    R = np.matrix([
        [sigma_i**2, 0.0],
        [0.0, sigma_j**2]
    ])

    return {
        "z": z_mat,
        "R": R,
        "sensor": sensor,
    }