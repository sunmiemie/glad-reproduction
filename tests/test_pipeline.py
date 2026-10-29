import numpy as np
from types import SimpleNamespace
from glad.types import Detection
from glad.pipeline import GLADPipeline, PipelineConfig, choose_detection
from glad.predictor import RelativeVelocityKalman, search_roi


class Appearance:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def detect(self, image, local):
        self.calls.append((local, image.shape))
        return next(self.results)


class Motion:
    def __init__(self, results=()):
        self.results = iter(results)
        self.calls = []
        self.last_diagnostics = {}

    def detect(self, previous, current, **kwargs):
        self.calls.append((previous.copy(), current.copy(), kwargs))
        return next(self.results, [])


def alignment(*args):
    return SimpleNamespace(H=np.eye(3), valid=True, inlier_ratio=1.)


def test_appearance_first_motion_fallback_and_coordinate_mapping():
    a = Appearance([[Detection((390, 290, 410, 310), .9, 'GAD')], []])
    m = Motion([[Detection((145, 145, 165, 165), .8, 'LMD')]])
    p = GLADPipeline(a, m, camera_estimator=alignment)
    frame = np.zeros((720, 1280, 3), np.uint8)
    r0 = p.process(frame)
    r1 = p.process(frame)
    assert r0.detection.source == 'GAD' and not r0.diagnostics['motion_called']
    assert r1.roi == (250, 150, 550, 450)
    assert r1.detection.box == (395, 295, 415, 315)
    assert len(m.calls) == 1 and a.calls[1][1][:2] == (300, 300)


def test_lost_frames_expand_roi_but_never_emit_prediction_as_detection():
    a = Appearance([[Detection((390, 290, 410, 310), .9, 'GAD')], [], [], []])
    p = GLADPipeline(a, Motion(), PipelineConfig(max_lost=2), alignment)
    frame = np.zeros((720, 1280, 3), np.uint8)
    results = [p.process(frame) for _ in range(4)]
    assert results[1].prediction is not None and results[1].detection is None
    assert results[2].roi[2] - results[2].roi[0] == 304
    assert results[3].mode == 'global' and results[3].roi is None


def test_crops_retain_size_at_boundary():
    assert search_roi((2, 2), 300, (720, 1280, 3)) == (0, 0, 300, 300)
    assert search_roi((1280, 720), 300, (720, 1280, 3)) == (980, 420, 1280, 720)
    assert search_roi((100, 50), 300, (100, 200, 3)) == (0, 0, 200, 100)


def test_camera_motion_is_not_learned_as_target_velocity():
    k = RelativeVelocityKalman((100, 100))
    H = np.array([[1, 0, 10], [0, 1, 3], [0, 0, 1]], dtype=float)
    for _ in range(5):
        pt = k.predict(H)
        k.update(pt)
    np.testing.assert_allclose(k.x, 0, atol=1e-8)
    np.testing.assert_allclose(k.center, (150, 115))


def test_velocity_learned_and_no_measurement_update_on_miss():
    k = RelativeVelocityKalman((100, 100))
    for t in range(1, 15):
        k.predict()
        k.update((100 + 3 * t, 100))
    pred = k.predict()
    assert abs(pred[0] - 145) < 1
    P = k.P.copy()
    k.predict()
    assert np.trace(k.P) > np.trace(P)


def test_ranking_is_order_independent_and_gated():
    ds = [Detection((100, 100, 110, 110), .99, 'LAD'),
          Detection((1, 1, 5, 5), .5, 'LAD'), Detection((3, 3, 7, 7), .4, 'LAD')]
    assert choose_detection(ds, (3, 3), 50) is ds[1]
    assert choose_detection(ds[::-1], (3, 3), 50) is ds[1]
