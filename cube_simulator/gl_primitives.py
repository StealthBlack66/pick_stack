"""저수준 OpenGL 헬퍼 (immediate mode).

GL 학습 목적에 맞추어 glBegin / glEnd 만 사용. PyOpenGL 그대로.
"""
from __future__ import annotations

import math
import numpy as np


# numpy 기반 행렬 -------------------------------------------------------------

def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


# 카메라 구좌표 → cartesian ----------------------------------------------------

def spherical_camera(target: np.ndarray,
                     distance: float,
                     yaw_deg: float,
                     pitch_deg: float) -> np.ndarray:
    """target 주변을 yaw/pitch/distance 로 도는 카메라 위치.

    pitch 90° = 정면 탑뷰 (z+ 위에서 내려다봄).
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    x = target[0] + distance * cp * math.cos(yaw)
    y = target[1] + distance * cp * math.sin(yaw)
    z = target[2] + distance * math.sin(pitch)
    return np.array([x, y, z], dtype=np.float64)


# GL 그리기 헬퍼 (PyOpenGL 가져오기는 lazy — import 시 GL 컨텍스트 없어도 OK)

def _gl():
    from OpenGL import GL  # noqa: WPS433
    return GL


def draw_cube(cx: float, cy: float, cz: float,
              size: float,
              yaw_deg: float = 0.0,
              color: tuple[float, float, float] = (0.4, 0.6, 0.9),
              edge_color: tuple[float, float, float] | None = (0.05, 0.05, 0.07)) -> None:
    """중심 (cx,cy,cz) 의 한 변 size 인 입방체.

    yaw_deg 는 z 축 기준 회전 (그리퍼 yaw 시각화용).
    """
    GL = _gl()
    GL.glPushMatrix()
    GL.glTranslatef(cx, cy, cz)
    GL.glRotatef(yaw_deg, 0.0, 0.0, 1.0)
    h = size * 0.5
    r, g, b = color
    GL.glColor3f(r, g, b)
    GL.glBegin(GL.GL_QUADS)
    # +X
    GL.glNormal3f(1, 0, 0)
    GL.glVertex3f(h, -h, -h); GL.glVertex3f(h, h, -h)
    GL.glVertex3f(h, h, h);   GL.glVertex3f(h, -h, h)
    # -X
    GL.glNormal3f(-1, 0, 0)
    GL.glVertex3f(-h, -h, -h); GL.glVertex3f(-h, -h, h)
    GL.glVertex3f(-h, h, h);   GL.glVertex3f(-h, h, -h)
    # +Y
    GL.glNormal3f(0, 1, 0)
    GL.glVertex3f(-h, h, -h); GL.glVertex3f(-h, h, h)
    GL.glVertex3f(h, h, h);   GL.glVertex3f(h, h, -h)
    # -Y
    GL.glNormal3f(0, -1, 0)
    GL.glVertex3f(-h, -h, -h); GL.glVertex3f(h, -h, -h)
    GL.glVertex3f(h, -h, h);   GL.glVertex3f(-h, -h, h)
    # +Z (위)
    GL.glNormal3f(0, 0, 1)
    GL.glColor3f(min(1.0, r * 1.15), min(1.0, g * 1.15), min(1.0, b * 1.15))
    GL.glVertex3f(-h, -h, h); GL.glVertex3f(h, -h, h)
    GL.glVertex3f(h, h, h);   GL.glVertex3f(-h, h, h)
    # -Z (아래)
    GL.glNormal3f(0, 0, -1)
    GL.glColor3f(r * 0.7, g * 0.7, b * 0.7)
    GL.glVertex3f(-h, -h, -h); GL.glVertex3f(-h, h, -h)
    GL.glVertex3f(h, h, -h);   GL.glVertex3f(h, -h, -h)
    GL.glEnd()

    if edge_color is not None:
        GL.glColor3f(*edge_color)
        GL.glLineWidth(1.5)
        _draw_cube_edges(h)
    GL.glPopMatrix()


def _draw_cube_edges(h: float) -> None:
    GL = _gl()
    pts = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h),  (h, -h, h),  (h, h, h),  (-h, h, h),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    GL.glBegin(GL.GL_LINES)
    for a, b in edges:
        GL.glVertex3f(*pts[a]); GL.glVertex3f(*pts[b])
    GL.glEnd()


def draw_grid(center_xy: tuple[float, float],
              pitch_mm: float,
              n: int,
              z: float,
              color: tuple[float, float, float] = (0.45, 0.45, 0.5)) -> None:
    """center_xy 를 (0,0) 그리드 셀 중심으로 두고 2n+1 칸 그리드 라인.

    셀의 외곽선이 아니라 셀 경계(셀 사이) 라인을 그림.
    """
    GL = _gl()
    GL.glColor3f(*color)
    GL.glLineWidth(1.0)
    half_n = n
    cx, cy = center_xy
    # x 방향 라인 (y 가변)
    GL.glBegin(GL.GL_LINES)
    for i in range(-half_n, half_n + 2):
        x0 = cx + (i - 0.5) * pitch_mm
        y_min = cy + (-half_n - 0.5) * pitch_mm
        y_max = cy + (half_n + 0.5) * pitch_mm
        GL.glVertex3f(x0, y_min, z)
        GL.glVertex3f(x0, y_max, z)
    for j in range(-half_n, half_n + 2):
        y0 = cy + (j - 0.5) * pitch_mm
        x_min = cx + (-half_n - 0.5) * pitch_mm
        x_max = cx + (half_n + 0.5) * pitch_mm
        GL.glVertex3f(x_min, y0, z)
        GL.glVertex3f(x_max, y0, z)
    GL.glEnd()


def draw_table(center_xy: tuple[float, float],
               pitch_mm: float,
               n: int,
               z: float,
               color: tuple[float, float, float] = (0.20, 0.22, 0.26)) -> None:
    """그리드 아래의 어두운 평면 (테이블)."""
    GL = _gl()
    cx, cy = center_xy
    pad = 1.0
    x_min = cx + (-n - 0.5 - pad) * pitch_mm
    x_max = cx + (n + 0.5 + pad) * pitch_mm
    y_min = cy + (-n - 0.5 - pad) * pitch_mm
    y_max = cy + (n + 0.5 + pad) * pitch_mm
    GL.glColor3f(*color)
    GL.glBegin(GL.GL_QUADS)
    GL.glVertex3f(x_min, y_min, z); GL.glVertex3f(x_max, y_min, z)
    GL.glVertex3f(x_max, y_max, z); GL.glVertex3f(x_min, y_max, z)
    GL.glEnd()


def draw_cube_ghost(cx: float, cy: float, cz: float,
                    size: float,
                    yaw_deg: float = 0.0,
                    color: tuple[float, float, float] = (0.4, 0.9, 0.6),
                    alpha: float = 0.30) -> None:
    """반투명 큐브 (호버 고스트 / 드래그 미리보기 용)."""
    GL = _gl()
    GL.glPushMatrix()
    GL.glTranslatef(cx, cy, cz)
    GL.glRotatef(yaw_deg, 0.0, 0.0, 1.0)
    h = size * 0.5
    r, g, b = color

    GL.glEnable(GL.GL_BLEND)
    GL.glDepthMask(GL.GL_FALSE)
    GL.glColor4f(r, g, b, alpha)
    GL.glBegin(GL.GL_QUADS)
    # 6면 — 단순화 (light shading 없음)
    faces = [
        ((h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)),     # +X
        ((-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)), # -X
        ((-h, h, -h), (-h, h, h), (h, h, h), (h, h, -h)),     # +Y
        ((-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)), # -Y
        ((-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)),     # +Z
        ((-h, -h, -h), (-h, h, -h), (h, h, -h), (h, -h, -h)), # -Z
    ]
    for f in faces:
        for v in f:
            GL.glVertex3f(*v)
    GL.glEnd()
    GL.glDepthMask(GL.GL_TRUE)

    # 윤곽선 강조 (alpha 무시, 완전 불투명)
    GL.glColor4f(r, g, b, 1.0)
    GL.glLineWidth(2.0)
    _draw_cube_edges(h)
    GL.glPopMatrix()


def draw_cube_outline(cx: float, cy: float, cz: float,
                      size: float,
                      yaw_deg: float = 0.0,
                      color: tuple[float, float, float] = (1.0, 0.6, 0.1),
                      line_width: float = 3.0) -> None:
    """큐브 윤곽선만 (선택 표시 용)."""
    GL = _gl()
    GL.glPushMatrix()
    GL.glTranslatef(cx, cy, cz)
    GL.glRotatef(yaw_deg, 0.0, 0.0, 1.0)
    h = size * 0.5
    GL.glColor3f(*color)
    GL.glLineWidth(line_width)
    _draw_cube_edges(h)
    GL.glLineWidth(1.0)
    GL.glPopMatrix()


def draw_gripper(cx: float, cy: float, cz: float,
                 yaw_deg: float = 0.0,
                 finger_width: float = 8.0,
                 finger_height: float = 60.0,
                 finger_depth: float = 14.0,
                 open_mm: float = 35.0,
                 body_color: tuple[float, float, float] = (0.85, 0.85, 0.90),
                 finger_color: tuple[float, float, float] = (0.30, 0.35, 0.45)) -> None:
    """단순한 parallel-jaw 그리퍼.

    (cx, cy, cz) = 그리퍼 끝 (두 finger 사이 중심). yaw_deg = z 축 회전.
    """
    GL = _gl()
    GL.glPushMatrix()
    GL.glTranslatef(cx, cy, cz)
    GL.glRotatef(yaw_deg, 0.0, 0.0, 1.0)

    # 본체 (직육면체) — finger 위쪽
    body_w = open_mm + finger_width * 2 + 12.0
    body_d = finger_depth + 10.0
    body_h = 40.0
    bx = body_w * 0.5; by = body_d * 0.5
    bz0 = finger_height
    bz1 = finger_height + body_h
    GL.glColor3f(*body_color)
    GL.glBegin(GL.GL_QUADS)
    # +X
    GL.glVertex3f(bx, -by, bz0); GL.glVertex3f(bx, by, bz0)
    GL.glVertex3f(bx, by, bz1);  GL.glVertex3f(bx, -by, bz1)
    # -X
    GL.glVertex3f(-bx, -by, bz0); GL.glVertex3f(-bx, -by, bz1)
    GL.glVertex3f(-bx, by, bz1);  GL.glVertex3f(-bx, by, bz0)
    # +Y
    GL.glVertex3f(-bx, by, bz0); GL.glVertex3f(-bx, by, bz1)
    GL.glVertex3f(bx, by, bz1);  GL.glVertex3f(bx, by, bz0)
    # -Y
    GL.glVertex3f(-bx, -by, bz0); GL.glVertex3f(bx, -by, bz0)
    GL.glVertex3f(bx, -by, bz1);  GL.glVertex3f(-bx, -by, bz1)
    # +Z / -Z (덜 중요)
    GL.glVertex3f(-bx, -by, bz1); GL.glVertex3f(bx, -by, bz1)
    GL.glVertex3f(bx, by, bz1);   GL.glVertex3f(-bx, by, bz1)
    GL.glVertex3f(-bx, -by, bz0); GL.glVertex3f(-bx, by, bz0)
    GL.glVertex3f(bx, by, bz0);   GL.glVertex3f(bx, -by, bz0)
    GL.glEnd()

    # 두 finger
    GL.glColor3f(*finger_color)
    half = open_mm * 0.5
    for side in (-1, 1):
        fx0 = side * half
        fx1 = fx0 + side * finger_width
        # x 정렬
        xmin = min(fx0, fx1); xmax = max(fx0, fx1)
        ymin = -finger_depth * 0.5; ymax = finger_depth * 0.5
        zmin = 0.0; zmax = finger_height
        GL.glBegin(GL.GL_QUADS)
        # 6면
        # +X
        GL.glVertex3f(xmax, ymin, zmin); GL.glVertex3f(xmax, ymax, zmin)
        GL.glVertex3f(xmax, ymax, zmax); GL.glVertex3f(xmax, ymin, zmax)
        # -X
        GL.glVertex3f(xmin, ymin, zmin); GL.glVertex3f(xmin, ymin, zmax)
        GL.glVertex3f(xmin, ymax, zmax); GL.glVertex3f(xmin, ymax, zmin)
        # +Y
        GL.glVertex3f(xmin, ymax, zmin); GL.glVertex3f(xmin, ymax, zmax)
        GL.glVertex3f(xmax, ymax, zmax); GL.glVertex3f(xmax, ymax, zmin)
        # -Y
        GL.glVertex3f(xmin, ymin, zmin); GL.glVertex3f(xmax, ymin, zmin)
        GL.glVertex3f(xmax, ymin, zmax); GL.glVertex3f(xmin, ymin, zmax)
        # +Z
        GL.glVertex3f(xmin, ymin, zmax); GL.glVertex3f(xmax, ymin, zmax)
        GL.glVertex3f(xmax, ymax, zmax); GL.glVertex3f(xmin, ymax, zmax)
        # -Z
        GL.glVertex3f(xmin, ymin, zmin); GL.glVertex3f(xmin, ymax, zmin)
        GL.glVertex3f(xmax, ymax, zmin); GL.glVertex3f(xmax, ymin, zmin)
        GL.glEnd()
    GL.glPopMatrix()


def draw_axes(origin: tuple[float, float, float], length: float) -> None:
    GL = _gl()
    ox, oy, oz = origin
    GL.glLineWidth(2.0)
    GL.glBegin(GL.GL_LINES)
    GL.glColor3f(1.0, 0.3, 0.3); GL.glVertex3f(ox, oy, oz); GL.glVertex3f(ox + length, oy, oz)
    GL.glColor3f(0.3, 1.0, 0.3); GL.glVertex3f(ox, oy, oz); GL.glVertex3f(ox, oy + length, oz)
    GL.glColor3f(0.3, 0.4, 1.0); GL.glVertex3f(ox, oy, oz); GL.glVertex3f(ox, oy, oz + length)
    GL.glEnd()
