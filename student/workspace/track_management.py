"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    
    # 1. Chuyển đổi vị trí từ cảm biến sang hệ tọa độ xe (vehicle frame)
    # p_v = R_s2v @ z + t_s2v
    R_s2v = meas.sensor.sens_to_veh[:3, :3]
    t_s2v = meas.sensor.sens_to_veh[:3, 3:]
    pos_v = R_s2v @ meas.z + t_s2v

    # 2. Vector trạng thái 6D (vị trí + vận tốc ban đầu bằng 0)
    x = np.matrix([
        [float(pos_v[0, 0])],
        [float(pos_v[1, 0])],
        [float(pos_v[2, 0])],
        [0.0],
        [0.0],
        [0.0],
    ])

    # 3. Ma trận hiệp phương sai P (6x6)
    # Khối vị trí (3x3): R xoay sang vehicle frame = R_s2v @ meas.R @ R_s2v.T
    P_pos = R_s2v @ meas.R @ R_s2v.T
    
    P = np.matrix(np.zeros((6, 6)))
    P[:3, :3] = P_pos
    P[3, 3] = params.sigma_p44**2
    P[4, 4] = params.sigma_p55**2
    P[5, 5] = params.sigma_p66**2

    # 4. Trạng thái và điểm ban đầu
    score = 1.0 / params.window
    state = "initialized"

    return {
        "x": x,
        "P": P,
        "state": state,
        "score": score,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    delta_score = 1.0 / params.window

    # Cập nhật điểm existence score
    if associated:
        track["score"] = min(1.0, track["score"] + delta_score)
    else:
        track["score"] -= delta_score

    # Cập nhật trạng thái vòng đời track
    if track["score"] > params.confirmed_threshold:
        track["state"] = "confirmed"
    elif track["state"] != "confirmed":
        # Giữ nguyên trạng thái confirmed khi bị miss; nếu chưa confirmed thì thành tentative
        track["state"] = "tentative"

    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()

    # 1. Phương sai vị trí nằm ngang (Pxx hoặc Pyy) vượt max_P
    if track["P"][0, 0] > params.max_P or track["P"][1, 1] > params.max_P:
        return True

    # 2. Điều kiện theo trạng thái
    if track["state"] == "confirmed":
        if track["score"] < params.delete_threshold:
            return True
    else:
        if track["score"] <= 0.0:
            return True

    return False