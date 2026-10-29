import cv2
import numpy as np
from glad.types import FrameResult, Detection
from glad.visualize import draw_result
from glad.__main__ import VideoSink


def test_small_video_preview_does_not_overflow():
    frame = np.zeros((64, 128, 3), np.uint8)
    result = FrameResult(0, 'global', Detection((20, 20, 40, 40), .9, 'GAD'),
                         None, None, 0, 1.)
    assert draw_result(frame, result).shape == frame.shape


def test_video_can_be_encoded_and_decoded(tmp_path):
    path = tmp_path / 'check.mp4'
    frame = np.full((64, 128, 3), 127, np.uint8)
    sink = VideoSink(path, 30, frame.shape)
    for _ in range(3):
        sink.write(frame)
    sink.close()
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        ok, decoded = cap.read()
        assert ok and decoded.shape == frame.shape
    finally:
        cap.release()
