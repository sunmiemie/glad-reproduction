"""Paper-inspired relative-velocity Kalman prediction (unpublished details reconstructed)."""
import numpy as np


def transform_point(point, H):
    q = np.asarray(H, dtype=float) @ np.array([point[0], point[1], 1.0])
    if not np.isfinite(q).all() or abs(q[2]) < 1e-8:
        return np.asarray(point, dtype=float).copy()
    return q[:2] / q[2]


def search_roi(center, side, shape):
    """Shift crops at edges to retain requested size, clipped to image dimensions."""
    h, w = shape[:2]
    rw, rh = min(int(side), w), min(int(side), h)
    x = int(np.clip(round(center[0] - rw / 2), 0, w - rw))
    y = int(np.clip(round(center[1] - rh / 2), 0, h - rh))
    return x, y, x + rw, y + rh


class RelativeVelocityKalman:
    """State [vx, vy, ax, ay], in pixels/frame; missing frames are predict-only.

    A camera-only observation anchor is accumulated through missing frames. On
    reacquisition the residual displacement / elapsed frames is an average
    velocity measurement, with increased measurement noise for longer gaps.
    Q, R, initial covariance and this gap update are engineering choices.
    """

    def __init__(self, center, process_noise=0.1, measurement_noise=4.0):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag([25., 25., 4., 4.])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.N = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.center = np.asarray(center, dtype=float)
        self.anchor = self.center.copy()
        self.steps = 0

    def predict(self, H=None):
        H = np.eye(3) if H is None else H
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.center = transform_point(self.center, H) + self.x[:2]
        self.anchor = transform_point(self.anchor, H)
        self.steps += 1
        return tuple(self.center)

    def update(self, center):
        center = np.asarray(center, dtype=float)
        if self.steps:
            z = (center - self.anchor) / self.steps
            R = self.R * self.steps
            S = self.N @ self.P @ self.N.T + R
            K = np.linalg.solve(S, self.N @ self.P).T
            self.x += K @ (z - self.N @ self.x)
            # Joseph update retains a positive covariance under roundoff.
            A = np.eye(4) - K @ self.N
            self.P = A @ self.P @ A.T + K @ R @ K.T
        self.center = center.copy()
        self.anchor = center.copy()
        self.steps = 0
