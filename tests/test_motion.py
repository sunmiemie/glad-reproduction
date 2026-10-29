import cv2
import numpy as np

from glad.motion import (Alignment, MotionDetector, estimate_camera_motion,
                         merge_boxes, motion_statistics)


def textured_scene(height=240, width=320):
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def test_camera_homography_maps_previous_into_current():
    first = textured_scene()
    true_H = np.array([[1., 0, 6], [0, 1, -4], [0, 0, 1]])
    second = cv2.warpPerspective(first, true_H, (320, 240))
    result = estimate_camera_motion(first, second)
    assert result.valid, result.reason
    point = np.array([[[160., 120.]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, result.H)[0, 0]
    np.testing.assert_allclose(transformed, [166, 116], atol=.5)
    assert result.inlier_ratio > .8


def test_textureless_alignment_is_invalid_and_motion_fails_closed():
    first = np.zeros((120, 160, 3), dtype=np.uint8)
    second = first.copy()
    cv2.rectangle(second, (40, 40), (65, 60), (255, 255, 255), -1)
    assert not estimate_camera_motion(first, second).valid
    detector = MotionDetector(lambda crop: 1.)
    assert detector.detect(first, second, local=False) == []
    assert detector.last_diagnostics["reason"] == "invalid_alignment"


def test_camera_translation_does_not_generate_candidates_from_border():
    first = textured_scene()
    H = np.array([[1., 0, 8], [0, 1, 5], [0, 0, 1]])
    second = cv2.warpPerspective(first, H, (320, 240))
    detector = MotionDetector(lambda crop: 1.)
    detections = detector.detect(first, second, local=False,
                                  alignment=Alignment(H, True, 1., np.hypot(8, 5)))
    assert detections == []
    assert detector.last_diagnostics["candidates"] == 0


def test_motion_variance_wraps_angles_and_rejects_inconsistent_motion():
    similar = np.array([[-2., .01], [-2., -.01], [-2., 0]])
    stats = motion_statistics(similar)
    assert stats["direction_variance"] < .001
    assert stats["magnitude_variance"] < .001
    opposing = motion_statistics([[2., 0], [-2., 0]])
    assert opposing["direction_variance"] > .8
    unequal = motion_statistics([[1., 0], [5., 0]])
    assert unequal["magnitude_variance"] > .8


def test_candidate_uses_current_crop_and_real_coherent_flow():
    first = np.full((160, 200, 3), 30, dtype=np.uint8)
    second = first.copy()
    cv2.rectangle(first, (80, 65), (100, 85), (220, 220, 220), -1)
    cv2.rectangle(second, (83, 68), (103, 88), (220, 220, 220), -1)
    captured = []
    def classifier(crop):
        captured.append(crop.copy())
        return .9
    detector = MotionDetector(classifier)
    found = detector.detect(first, second, local=True,
                            alignment=Alignment(np.eye(3), True, 1., 0.))
    assert found, detector.last_diagnostics
    assert found[0].source == "LMD"
    box = tuple(int(v) for v in found[0].box)
    np.testing.assert_array_equal(captured[0], second[box[1]:box[3], box[0]:box[2]])
    assert detector.last_diagnostics["motion_passed"] >= 1


def test_nearby_motion_components_merge_transitively():
    assert merge_boxes([(0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)], 15) == [(0, 0, 50, 10)]
