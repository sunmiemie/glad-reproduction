"""Appearance-first global/local GLAD state machine with motion fallback."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .types import FrameResult
from .predictor import RelativeVelocityKalman, search_roi


@dataclass
class PipelineConfig:
    roi_size: int = 300
    roi_growth: int = 4
    max_lost: int = 30
    association_radius: float = 50.0
    prediction: bool = True
    motion: bool = True
    global_only: bool = False

    def __post_init__(self):
        if self.roi_size < 32 or self.roi_growth < 0 or self.max_lost < 1:
            raise ValueError('roi_size >= 32, roi_growth >= 0, max_lost >= 1 required')
        if self.association_radius <= 0:
            raise ValueError('association_radius must be positive')


def choose_detection(detections, center=None, radius=None):
    """Deterministic confidence + distance ranking, unlike upstream order dependence."""
    valid = [d for d in detections if np.isfinite(d.box).all()
             and np.isfinite(d.confidence) and 0 <= d.confidence <= 1
             and d.box[2] > d.box[0] and d.box[3] > d.box[1]]
    if center is None:
        return min(valid, key=lambda d: (-d.confidence, tuple(d.box))) if valid else None
    distances = {id(d): float(np.linalg.norm(np.array(d.center) - center)) for d in valid}
    if radius is not None:
        valid = [d for d in valid if distances[id(d)] <= radius]
    by_conf = sorted(valid, key=lambda d: (-d.confidence, distances[id(d)], tuple(d.box)))
    by_dist = sorted(valid, key=lambda d: (distances[id(d)], -d.confidence, tuple(d.box)))
    rank = {id(d): i for i, d in enumerate(by_conf)}
    for i, d in enumerate(by_dist):
        rank[id(d)] += i
    return min(valid, key=lambda d: (rank[id(d)], -d.confidence,
                                    distances[id(d)], tuple(d.box))) if valid else None


class GLADPipeline:
    def __init__(self, appearance, motion_detector, config=None, camera_estimator=None):
        self.appearance = appearance
        self.motion_detector = motion_detector
        self.config = config or PipelineConfig()
        if camera_estimator is None:
            from .motion import estimate_camera_motion
            camera_estimator = estimate_camera_motion
        self.camera_estimator = camera_estimator
        self.previous = None
        self.predictor = None
        self.lost = 0
        self.index = 0

    def reset(self):
        self.previous = None
        self.predictor = None
        self.lost = 0
        self.index = 0

    def process(self, frame):
        start = perf_counter()
        if self.previous is not None and frame.shape != self.previous.shape:
            raise ValueError('Video frame dimensions changed; reset the pipeline first')
        local = self.predictor is not None and not self.config.global_only
        mode = 'local' if local else 'global'
        roi = prediction = alignment = None
        diagnostics = {'appearance_called': True, 'motion_called': False}
        image, previous = frame, self.previous
        if local:
            if self.config.prediction:
                alignment = self.camera_estimator(self.previous, frame)
                prediction = self.predictor.predict(alignment.H if alignment.valid else None)
                diagnostics['camera_valid'] = bool(alignment.valid)
                diagnostics['camera_inlier_ratio'] = float(alignment.inlier_ratio)
            else:
                prediction = tuple(self.predictor.center)
            size = self.config.roi_size + (self.config.roi_growth * self.lost
                                           if self.config.prediction else 0)
            roi = search_roi(prediction, size, frame.shape)
            x1, y1, x2, y2 = roi
            image, previous = frame[y1:y2, x1:x2], self.previous[y1:y2, x1:x2]
            center = (prediction[0] - x1, prediction[1] - y1)
            radius = self.config.association_radius + self.config.roi_growth * self.lost
        else:
            center = radius = None
        candidates = self.appearance.detect(image, local=local)
        diagnostics['appearance_candidates'] = len(candidates)
        detection = choose_detection(candidates, center, radius)
        if detection is None and self.config.motion and previous is not None:
            diagnostics['motion_called'] = True
            # Local crops use the SAME current-frame rectangle in both images.
            candidates = self.motion_detector.detect(previous, image, local=local,
                                                      expected_center=center)
            diagnostics['motion'] = dict(self.motion_detector.last_diagnostics)
            detection = choose_detection(candidates, center, radius)
        if detection is not None:
            if roi is not None:
                detection = detection.translated(roi[0], roi[1])
            self.lost = 0
            if self.predictor is None:
                self.predictor = RelativeVelocityKalman(detection.center)
            else:
                self.predictor.update(detection.center)
        elif local:
            self.lost += 1
            if self.lost >= self.config.max_lost:
                self.predictor = None
                diagnostics['next_mode'] = 'global'
        if self.config.global_only:
            self.predictor = None
        self.previous = frame.copy()
        result = FrameResult(self.index, mode, detection, roi, prediction,
                             self.lost, (perf_counter() - start) * 1000, diagnostics)
        self.index += 1
        return result
