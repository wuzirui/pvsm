import math
import numpy as np
import torch


def _rand_unit_quat_np():
    q = np.random.normal(size=4).astype(np.float64)
    n = np.linalg.norm(q) + 1e-12
    q = q / n
    return q  # w, x, y, z


def _quat_to_R_np(q):
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def _rand_translation_np(max_norm=2.0):
    d = np.random.normal(size=3).astype(np.float64)
    n = np.linalg.norm(d) + 1e-12
    d = d / n
    r = max_norm * (np.random.rand() ** (1.0 / 3.0))
    return (r * d).astype(np.float32)


def _rand_axis_np():
    d = np.random.normal(size=3).astype(np.float64)
    n = np.linalg.norm(d) + 1e-12
    d = d / n
    return d.astype(np.float32)


def _rand_se3_np(angle_deg, max_t_norm=2.0):
    angle_rad = angle_deg * math.pi / 180.0

    axis = _rand_axis_np()
    s = math.sin(angle_rad / 2.0)
    q = np.array(
        [
            math.cos(angle_rad / 2.0),
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
        ],
        dtype=np.float32,
    )

    R = _quat_to_R_np(q)
    t = _rand_translation_np(max_t_norm)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    print(f"translation: {t}, angle: {angle_deg:.2f}°, axis: {axis}")
    return T
