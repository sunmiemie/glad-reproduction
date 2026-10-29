"""Video overlays: actual detections and ROI zoom, without trajectory trails."""
import cv2


def draw_result(frame, result):
    canvas = frame.copy()
    h, w = frame.shape[:2]
    d = result.detection
    if result.roi:
        x1, y1, x2, y2 = result.roi
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (220, 160, 40), 1)
    if d:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), d.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 255), 2)
        cv2.putText(canvas, '%s %.2f' % (d.source, d.confidence), (max(0, x1), max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 230, 255), 2, cv2.LINE_AA)
        from .predictor import search_roi
        roi = result.roi or search_roi(d.center, 300, frame.shape)
        a, b, c, e = roi
        crop = frame[b:e, a:c].copy()
        cv2.rectangle(crop, (x1 - a, y1 - b), (x2 - a, y2 - b), (0, 230, 255), 2)
        side = min(320, h // 2, w // 3, h - 50)
        if side >= 32:
            zoom = cv2.resize(crop, (side, side), interpolation=cv2.INTER_LINEAR)
            ox, oy = w - side - 10, 40
            canvas[oy:oy + side, ox:ox + side] = zoom
            cv2.rectangle(canvas, (ox, oy), (ox + side, oy + side), (220, 160, 40), 2)
            label = ('DETECTION ROI' if result.roi else 'TARGET PREVIEW')
            cv2.putText(canvas, label, (ox, oy - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (255, 255, 255), 1, cv2.LINE_AA)
    line = 'GLAD | frame %d | %s | %s | lost %d | %.1f ms' % (
        result.index, result.mode.upper(), d.source if d else 'NO DETECTION',
        result.lost_frames, result.elapsed_ms)
    cv2.rectangle(canvas, (0, 0), (min(w, 840), 30), (20, 20, 20), -1)
    cv2.putText(canvas, line, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .53,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
