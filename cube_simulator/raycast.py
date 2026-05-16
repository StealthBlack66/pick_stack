"""화면 픽셀 → 월드 ray, ray-plane intersect.

좌표계: OpenGL 표준
- NDC: x [-1, +1], y [-1, +1] (위로 +), z [-1, +1]
- 클립 → 카메라 → 월드 는 inv(proj·view) 한 번에 처리.
"""
from __future__ import annotations

import numpy as np


def screen_to_world_ray(mouse_x: int, mouse_y: int,
                        viewport_w: int, viewport_h: int,
                        proj: np.ndarray,
                        view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """픽셀 좌표 → (origin, direction) 월드 ray.

    mouse_y 는 Qt 좌표 (위가 0). NDC y 는 위가 +1 이라 뒤집어야 함.
    """
    # NDC
    ndc_x = (2.0 * mouse_x / viewport_w) - 1.0
    ndc_y = 1.0 - (2.0 * mouse_y / viewport_h)
    # near / far 점을 NDC에서 픽업 → 월드로 unproject
    inv = np.linalg.inv(proj @ view)
    near_h = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
    far_h = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64)
    near = near_h[:3] / near_h[3]
    far = far_h[:3] / far_h[3]
    direction = far - near
    n = np.linalg.norm(direction)
    if n > 1e-9:
        direction /= n
    return near, direction


def ray_plane_z(origin: np.ndarray, direction: np.ndarray, z_plane: float) -> np.ndarray | None:
    """수평 평면 z = z_plane 과의 교차점.

    direction.z 가 0 (평면과 평행) 이면 None.
    카메라 뒤쪽으로 가는 교차도 None (t < 0).
    """
    if abs(direction[2]) < 1e-9:
        return None
    t = (z_plane - origin[2]) / direction[2]
    if t < 0:
        return None
    return origin + t * direction


def ray_obb_z(origin: np.ndarray,
              direction: np.ndarray,
              center: tuple[float, float, float],
              half_extents: tuple[float, float, float],
              yaw_deg: float) -> float | None:
    """ray ↔ z 축으로 yaw 회전된 직육면체 OBB 의 첫 교차 t.

    half_extents = (hx, hy, hz). 반환값은 ray origin 에서 hit 까지의 t.
    miss / 카메라 뒤쪽 hit 이면 None.

    slab method 를 큐브의 local frame (yaw 만큼 역회전) 에서 수행.
    """
    import math
    yaw = math.radians(yaw_deg)
    cs = math.cos(yaw)
    sn = math.sin(yaw)
    # 카메라 → 큐브 local 변환 (yaw 역회전)
    dx0 = origin[0] - center[0]
    dy0 = origin[1] - center[1]
    dz0 = origin[2] - center[2]
    lo_x = cs * dx0 + sn * dy0
    lo_y = -sn * dx0 + cs * dy0
    lo_z = dz0
    ld_x = cs * direction[0] + sn * direction[1]
    ld_y = -sn * direction[0] + cs * direction[1]
    ld_z = direction[2]

    t_min = -float('inf')
    t_max = float('inf')
    EPS = 1e-9
    for o, d, h in (
        (lo_x, ld_x, half_extents[0]),
        (lo_y, ld_y, half_extents[1]),
        (lo_z, ld_z, half_extents[2]),
    ):
        if abs(d) < EPS:
            if o < -h or o > h:
                return None
            continue
        t1 = (-h - o) / d
        t2 = (h - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
        if t2 < t_max:
            t_max = t2
        if t_min > t_max:
            return None
    if t_max < 0:
        return None
    return max(0.0, t_min)
