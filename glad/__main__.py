import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
import cv2
import numpy as np
from .pipeline import GLADPipeline, PipelineConfig
from .visualize import draw_result

ROOT = Path(__file__).resolve().parents[1]


def json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


class VideoSink:
    def __init__(self, path, fps, shape):
        h, w = shape[:2]
        exe = shutil.which('ffmpeg')
        if not exe:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass
        self.process = self.writer = self.log = None
        if exe:
            self.log = open(path.with_suffix('.ffmpeg.log'), 'w', encoding='utf-8')
            self.process = subprocess.Popen([
                exe, '-hide_banner', '-loglevel', 'warning', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', '%dx%d' % (w, h), '-r', str(fps), '-i', 'pipe:0',
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '21',
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
            self.codec = 'H.264'
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError('Unable to open output video writer')
            self.codec = 'mp4v'

    def write(self, frame):
        if self.process:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
                code = self.process.wait(timeout=60)
            except (subprocess.TimeoutExpired, BrokenPipeError):
                self.process.kill()
                self.process.wait(timeout=10)
                raise
            finally:
                self.log.close()
            if code:
                raise RuntimeError('FFmpeg failed; see output.ffmpeg.log')
        elif self.writer:
            self.writer.release()


def parser():
    p = argparse.ArgumentParser(description='GLAD: official appearance weights + reconstructed adaptive search')
    p.add_argument('--source', type=Path, required=True, help='Local video path')
    p.add_argument('--output', type=Path, default=ROOT / 'runs' / 'glad')
    p.add_argument('--weights', type=Path, default=ROOT / 'weights')
    p.add_argument('--yolo-root', type=Path, default=ROOT / 'external' / 'yolov5')
    p.add_argument('--device', default='cpu', help='cpu or cuda:0; no automatic AMD GPU backend')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--roi-size', type=int, default=300)
    p.add_argument('--roi-growth', type=int, default=4)
    p.add_argument('--max-lost', type=int, default=30)
    p.add_argument('--association-radius', type=float, default=50)
    p.add_argument('--global-conf', type=float, default=.5)
    p.add_argument('--local-conf', type=float, default=.1)
    p.add_argument('--max-frames', type=int, default=0, help='0 = entire video')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--fixed-roi', action='store_true', help='Disable prediction and ROI growth for ablation')
    p.add_argument('--no-motion', action='store_true')
    p.add_argument('--global-only', action='store_true')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--display', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if not args.source.is_file():
        raise SystemExit('Video does not exist: %s' % args.source)
    if args.max_frames < 0 or args.start_frame < 0 or args.threads < 1:
        raise SystemExit('Frame limits must be nonnegative and threads must be positive')
    if args.imgsz < 32 or args.imgsz % 32:
        raise SystemExit('imgsz must be a positive multiple of 32')
    if not 0 < args.global_conf <= 1 or not 0 < args.local_conf <= 1:
        raise SystemExit('Confidence must be in (0, 1]')
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output directory is not empty; choose a new --output')
    config = PipelineConfig(args.roi_size, args.roi_growth, args.max_lost,
                            args.association_radius, not args.fixed_roi,
                            not args.no_motion, args.global_only)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(args.threads)
    from .appearance import AppearanceDetector
    from .motion import MotionDetector
    import torch
    load_start = perf_counter()
    detector = AppearanceDetector(
        args.weights / 'yolov5s_GLAD.pt', args.weights / 'yolov5s_GLAD-crop.pt',
        args.weights / 'Net_best.pth', args.yolo_root, device=args.device,
        imgsz=args.imgsz, threads=args.threads,
        global_conf=args.global_conf, local_conf=args.local_conf)
    # Warm both inference shapes; initialization is reported separately.
    blank = np.zeros((args.imgsz, args.imgsz, 3), np.uint8)
    detector.detect(blank, local=False)
    detector.detect(blank, local=True)
    detector.classify(blank[:32, :32])
    setup_seconds = perf_counter() - load_start
    pipeline = GLADPipeline(detector, MotionDetector(detector.classify), config)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit('OpenCV could not open video')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    sink = None
    frames, times, modes = Counter(), [], Counter()
    completed = False
    reached_eof = False
    stopped_by_user = False
    saved_preview = False
    wall_start = perf_counter()
    print('Running %s on %s; output %s' % (args.source.name, args.device, args.output), flush=True)
    try:
        with (args.output / 'frames.jsonl').open('w', encoding='utf-8') as log:
            while not args.max_frames or len(times) < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    completed = True
                    reached_eof = True
                    break
                result = pipeline.process(frame)
                row = asdict(result)
                row['frame_index'] = result.index + args.start_frame
                row['time_seconds'] = row['frame_index'] / source_fps
                log.write(json.dumps(row, default=json_default) + '\n')
                source = result.detection.source if result.detection else 'miss'
                frames[source] += 1
                modes[result.mode] += 1
                times.append(result.elapsed_ms)
                rendered = draw_result(frame, result)
                if not saved_preview and result.detection and result.index >= 10:
                    cv2.imwrite(str(args.output / 'preview.jpg'), rendered)
                    saved_preview = True
                if not args.no_video:
                    if sink is None:
                        sink = VideoSink(args.output / 'output.mp4', source_fps, frame.shape)
                    sink.write(rendered)
                if args.display:
                    cv2.imshow('GLAD reproduction - q to stop', rendered)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        stopped_by_user = True
                        break
                if len(times) % 60 == 0:
                    print('frames=%d detected=%d pipeline_fps=%.2f' % (
                        len(times), len(times) - frames['miss'], 1000 / np.mean(times)), flush=True)
            else:
                completed = True
    finally:
        cap.release()
        finalize_error = None
        if sink:
            try:
                sink.close()
            except Exception as exc:
                finalize_error = '%s: %s' % (type(exc).__name__, exc)
                completed = False
        if args.display:
            cv2.destroyAllWindows()
        elapsed = perf_counter() - wall_start
        n = len(times)
        summary = {
            'source': str(args.source.resolve()), 'source_reported_frames': total,
            'reached_decoder_eof': reached_eof,
            'eof_before_reported_count': bool(reached_eof and total > 0 and n + args.start_frame < total),
            'frame_count_note': 'Reported count comes from container metadata; processed_frames counts decoded images. An early EOF requires source verification.',
            'source_fps': source_fps, 'start_frame': args.start_frame,
            'processed_frames': n, 'completed_requested_run': completed,
            'stopped_by_user': stopped_by_user, 'branches': dict(frames), 'modes': dict(modes),
            'detection_coverage': (n - frames['miss']) / n if n else 0,
            'coverage_warning': 'Coverage counts model outputs, NOT accuracy, recall or AP; no ground truth evaluated.',
            'pipeline_fps': 1000 / np.mean(times) if n else 0,
            'pipeline_ms_mean': float(np.mean(times)) if n else 0,
            'pipeline_ms_p95': float(np.percentile(times, 95)) if n else 0,
            'end_to_end_fps': n / elapsed if elapsed else 0,
            'wall_seconds': elapsed, 'model_setup_and_warmup_seconds': setup_seconds,
            'video_codec': sink.codec if sink else None,
            'video_finalize_error': finalize_error, 'device': args.device,
            'torch': torch.__version__, 'opencv': cv2.__version__, 'python': platform.python_version(),
            'platform': platform.platform(), 'threads': args.threads,
            'config': asdict(config), 'imgsz': args.imgsz,
            'global_conf': args.global_conf, 'local_conf': args.local_conf,
        }
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, default=json_default), encoding='utf-8')
        print(json.dumps(summary, indent=2, default=json_default), flush=True)
    if finalize_error:
        raise SystemExit('Video finalization failed: ' + finalize_error)
    if not times:
        raise SystemExit('No frames decoded')


if __name__ == '__main__':
    main()
