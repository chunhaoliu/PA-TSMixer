"""History-only motion groups of the test windows (Section V-A), used by the motion and scale analyses.

Vertical: least-squares altitude slope over the last 11 observed points; > 1 m/s climb, < -1 m/s descent, else level.
Horizontal: displacements over the last two 10-s intervals; if both speeds exceed 5 m/s, a heading change above
10 degrees is turning and the rest non-turning; other windows are low speed.
"""
import numpy as np

EXPECTED = {"trajair": dict(climb=5311, descent=5325, level=10710, turning=5893, nonturning=11901, low_speed=3552),
            "tartanaviation": dict(climb=11645, descent=13398, level=31966, turning=12172, nonturning=31698,
                                   low_speed=13139)}


def groups(history):
    """history: [N, 60, 3] metres -> dict of boolean masks."""
    t = np.arange(11, dtype=np.float64) - 5.
    slope = (history[:, -11:, 2] - history[:, -11:, 2].mean(axis=1, keepdims=True)) @ t / np.sum(t * t)
    d1 = history[:, -11, :2] - history[:, -21, :2]
    d2 = history[:, -1, :2] - history[:, -11, :2]
    s1, s2 = np.linalg.norm(d1, axis=1) / 10, np.linalg.norm(d2, axis=1) / 10
    moving = (s1 > 5) & (s2 > 5)
    cos = np.sum(d1 * d2, axis=1) / np.maximum(s1 * s2 * 100, 1e-12)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return dict(climb=slope > 1, descent=slope < -1, level=np.abs(slope) <= 1,
                turning=moving & (angle > 10), nonturning=moving & (angle <= 10), low_speed=~moving)
