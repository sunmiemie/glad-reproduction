"""Camera-compensated motion branch for the GLAD reconstruction.

The publication describes direction/magnitude variances, unlike the unstable
signed coefficients of variation in the public demo. We use radians squared
for direction variance (wrapped about the circular mean), pixels squared for
magnitude variance, and the reported 0.8 thresholds. The unpublished global
threshold coefficients default to alpha=1, beta=0: these are engineering
choices, not recovered author settings. Local threshold 4 follows the demo.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .types import Detection


@dataclass
class Alignment:
    """H maps previous-frame points into current-frame coordinates."""

    H: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    valid: bool = False
    inlier_ratio: float = 0.0
    mean_displacement: float = 0.0
    reason: str = "unavailable"
    matched_points: int = 0


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def estimate_camera_motion(previous, current) -> Alignment:
    """Estimate previous-to-current homography with a 30 x 20 LK grid.

    Forward/backward consistency, minimum support, and transformed-corner
    checks prevent failed fits from producing a field of motion detections.
    Low-texture scenes return an explicitly invalid identity transform.
    """
    if previous is None or current is None or previous.shape != current.shape:
        return Alignment(reason="missing_or_mismatched_frames")
    height, width = current.shape[:2]
    if min(height, width) < 24:
        return Alignment(reason="image_too_small")
    previous_gray, current_gray = _gray(previous), _gray(current)
    xs = np.linspace(8, width - 9, 30, dtype=np.float32)
    ys = np.linspace(8, height - 9, 20, dtype=np.float32)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)
    params = dict(winSize=(15, 15), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    try:
        tracked, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None, **params)
        if tracked is None or forward_status is None:
            return Alignment(reason="lk_failed")
        reverse, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, tracked, None, **params)
        if reverse is None or backward_status is None:
            return Alignment(reason="backward_lk_failed")
        source, target = points[:, 0], tracked[:, 0]
        flow = np.linalg.norm(target - source, axis=1)
        good = (forward_status.ravel() == 1) & (backward_status.ravel() == 1)
        good &= np.isfinite(target).all(axis=1) & np.isfinite(reverse[:, 0]).all(axis=1)
        good &= np.linalg.norm(reverse[:, 0] - source, axis=1) < 1.5
        good &= (target[:, 0] >= 0) & (target[:, 0] < width)
        good &= (target[:, 1] >= 0) & (target[:, 1] < height)
        good &= flow < .25 * np.hypot(width, height)
        count = int(good.sum())
        if count < 12:
            return Alignment(reason="insufficient_lk_support", matched_points=count)
        H, inliers = cv2.findHomography(source[good], target[good], cv2.RANSAC, 3.0)
        ratio = float(inliers.mean()) if inliers is not None else 0.0
        displacement = float(flow[good].mean())
        if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-9:
            return Alignment(reason="homography_failed", matched_points=count)
        H = H / H[2, 2]
        corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                            [0, height - 1]], dtype=np.float32)
        projected = cv2.perspectiveTransform(corners[None], H)[0]
        area_ratio = abs(cv2.contourArea(projected)) / ((width - 1) * (height - 1))
        plausible = (np.isfinite(projected).all() and cv2.isContourConvex(projected)
                     and .25 < area_ratio < 4.0
                     and np.max(np.abs(projected - corners)) < max(width, height))
        if ratio < .5 or not plausible or abs(np.linalg.det(H)) < 1e-8:
            return Alignment(inlier_ratio=ratio, mean_displacement=displacement,
                             reason="unreliable_homography", matched_points=count)
        return Alignment(H, True, ratio, displacement, "ok", count)
    except cv2.error:
        return Alignment(reason="opencv_alignment_failed")


def motion_statistics(vectors):
    """Return actual variances without a signed/near-zero angle denominator."""
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    vectors = vectors[np.isfinite(vectors).all(axis=1)]
    if len(vectors) == 0:
        return None
    magnitudes = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    center = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    relative = np.arctan2(np.sin(angles - center), np.cos(angles - center))
    return {"mean_magnitude": float(magnitudes.mean()),
            "magnitude_variance": float(magnitudes.var()),
            "direction_variance": float(relative.var()),
            "points": len(vectors)}


def _rectangle_distance(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return np.hypot(dx, dy)


def merge_boxes(boxes, distance=15):
    """Merge nearby motion blobs, with deterministic transitive unions."""
    remaining = sorted(tuple(int(v) for v in box) for box in boxes)
    merged = []
    while remaining:
        box = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for index, other in enumerate(remaining):
                if _rectangle_distance(box, other) < distance:
                    box = (min(box[0], other[0]), min(box[1], other[1]),
                           max(box[2], other[2]), max(box[3], other[3]))
                    remaining.pop(index)
                    changed = True
                    break
        merged.append(box)
    return merged


class MotionDetector:
    def __init__(self, classifier: Callable, alpha=1.0, beta=0.0,
                 classifier_threshold=.5, merge_distance=15,
                 direction_variance_threshold=.8, magnitude_variance_threshold=.8):
        self.classifier = classifier
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.classifier_threshold = float(classifier_threshold)
        self.merge_distance = float(merge_distance)
        self.direction_variance_threshold = float(direction_variance_threshold)
        self.magnitude_variance_threshold = float(magnitude_variance_threshold)
        self.last_diagnostics = {}

    @staticmethod
    def _candidate_vectors(previous_gray, current_gray, box):
        x1, y1, x2, y2 = box
        # Context around the component avoids artificial crop-boundary flow.
        height, width = current_gray.shape
        ax, ay, bx, by = max(0, x1 - 8), max(0, y1 - 8), min(width, x2 + 8), min(height, y2 + 8)
        source = previous_gray[ay:by, ax:bx]
        target = current_gray[ay:by, ax:bx]
        feature_mask = np.zeros(source.shape, dtype=np.uint8)
        feature_mask[y1 - ay:y2 - ay, x1 - ax:x2 - ax] = 255
        points = cv2.goodFeaturesToTrack(source, maxCorners=30, qualityLevel=.15,
                                       minDistance=3, blockSize=3, mask=feature_mask)
        if points is None:
            return None
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            source, target, points, None, winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .03))
        if tracked is None or status is None:
            return None
        good = status.ravel() == 1
        good &= np.isfinite(tracked[:, 0]).all(axis=1)
        good &= (tracked[:, 0, 0] >= 0) & (tracked[:, 0, 0] < source.shape[1])
        good &= (tracked[:, 0, 1] >= 0) & (tracked[:, 0, 1] < source.shape[0])
        return tracked[good, 0] - points[good, 0]

    def detect(self, previous, current, local: bool,
               alignment: Optional[Alignment] = None, expected_center=None):
        self.last_diagnostics = {"candidates": 0, "motion_passed": 0, "accepted": 0,
                                 "rejected_no_flow": 0, "rejected_motion": 0,
                                 "rejected_appearance": 0}
        if previous is None or current is None or previous.shape != current.shape:
            self.last_diagnostics["reason"] = "missing_or_mismatched_frames"
            return []
        alignment = estimate_camera_motion(previous, current) if alignment is None else alignment
        self.last_diagnostics.update(alignment_valid=bool(alignment.valid),
                                     alignment_reason=alignment.reason,
                                     inlier_ratio=float(alignment.inlier_ratio))
        if not alignment.valid or not np.isfinite(alignment.H).all():
            self.last_diagnostics["reason"] = "invalid_alignment"
            return []
        height, width = current.shape[:2]
        prev_gray = cv2.GaussianBlur(_gray(previous), (11, 11), 0)
        curr_gray = cv2.GaussianBlur(_gray(current), (11, 11), 0)
        compensated = cv2.warpPerspective(prev_gray, alignment.H, (width, height))
        valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8),
                                    alignment.H, (width, height), flags=cv2.INTER_NEAREST)
        # Remove interpolation and blur support around the image boundary.
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        difference = cv2.absdiff(curr_gray, compensated)
        brightness_change = float(cv2.absdiff(prev_gray, curr_gray).mean())
        threshold = 4.0 if local else 5.0 + self.alpha * brightness_change + self.beta * alignment.mean_displacement
        binary = np.where((difference > threshold) & (valid > 0), 255, 0).astype(np.uint8)
        if not local:
            binary = cv2.medianBlur(binary, 5)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary[valid == 0] = 0
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_diagnostics.update(threshold=float(threshold), raw_components=len(contours))
        if len(contours) > 500:
            self.last_diagnostics["reason"] = "too_many_raw_components"
            return []
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 30 < cv2.contourArea(contour) < 3000 and (.5 if local else .6) < w / h < 3:
                boxes.append((x, y, x + w, y + h))
        if self.merge_distance > 0:
            boxes = merge_boxes(boxes, self.merge_distance)
        self.last_diagnostics["candidates"] = len(boxes)
        if len(boxes) > (30 if local else 50):
            self.last_diagnostics["reason"] = "too_many_candidates"
            return []
        accepted = []
        for x1, y1, x2, y2 in boxes:
            box = (max(0, x1 - 2), max(0, y1 - 2), min(width, x2 + 2), min(height, y2 + 2))
            x1, y1, x2, y2 = box
            if not .5 < (x2 - x1) / (y2 - y1) < 3 or (x2 - x1) * (y2 - y1) > 6000:
                continue
            try:
                vectors = self._candidate_vectors(compensated, curr_gray, box)
            except cv2.error:
                vectors = None
            stats = motion_statistics(vectors) if vectors is not None else None
            if stats is None:
                self.last_diagnostics["rejected_no_flow"] += 1
                continue
            if (stats["mean_magnitude"] < (.6 if local else 1.0)
                    or stats["direction_variance"] > self.direction_variance_threshold
                    or stats["magnitude_variance"] > self.magnitude_variance_threshold):
                self.last_diagnostics["rejected_motion"] += 1
                continue
            self.last_diagnostics["motion_passed"] += 1
            confidence = float(self.classifier(current[y1:y2, x1:x2]))
            if not np.isfinite(confidence) or confidence <= self.classifier_threshold:
                self.last_diagnostics["rejected_appearance"] += 1
                continue
            accepted.append(Detection(tuple(float(v) for v in box), confidence,
                                      "LMD" if local else "GMD"))
        if expected_center is not None:
            accepted.sort(key=lambda d: np.hypot(d.center[0] - expected_center[0],
                                                  d.center[1] - expected_center[1]))
        else:
            accepted.sort(key=lambda d: -d.confidence)
        self.last_diagnostics.update(accepted=len(accepted), reason="ok")
        return accepted
