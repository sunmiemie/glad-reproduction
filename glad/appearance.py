"""Published GLAD checkpoints on a portable PyTorch backend.

YOLO architecture code is fetched separately from pinned YOLOv5 v6.0. The
checkpoint-compatible classifier below follows the *used* Net in the author's
Functions.py; the unused MyNet class is a different network.
"""
import importlib
import os
import pathlib
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from .types import Detection


class CandidateNet(nn.Module):
    """Architecture matching the published Net_best.pth (two logits)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def letterbox(image, size=640):
    """Return RGB CHW tensor, actual xy scale, integer xy padding.

    Pad value 128 matches GLAD's published TensorRT preprocessing. Separate
    scale factors account for integer rounding of non-square input dimensions.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Expected a nonempty BGR image with three channels")
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    rw, rh = max(1, int(w * ratio)), max(1, int(h * ratio))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    left, top = (size - rw) // 2, (size - rh) // 2
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    canvas[top:top + rh, left:left + rw] = resized
    rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).float().div_(255).unsqueeze(0), (rw / w, rh / h), (left, top)


def restore_boxes(boxes, scale, pad, shape):
    """Map network xyxy coordinates into the original crop, without rounding."""
    boxes = boxes.clone()
    h, w = shape[:2]
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad[0]) / scale[0]
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad[1]) / scale[1]
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, w)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, h)
    return boxes


@contextmanager
def _checkpoint_paths():
    # Legacy Linux checkpoints may contain PosixPath metadata. Limit this
    # compatibility patch to trusted checkpoint deserialization.
    previous = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = previous


def _load_yolo(path, device):
    with _checkpoint_paths():
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float().to(device).eval()
    # YOLOv5 v6.0's Detect expects a list, while older saved models may store a
    # registered tensor buffer. This is the compatibility fix in attempt_load.
    from models.yolo import Detect
    for module in model.modules():
        if isinstance(module, Detect):
            module.inplace = True
            if not isinstance(module.anchor_grid, list):
                delattr(module, "anchor_grid")
                module.anchor_grid = [torch.zeros(1, device=device)] * module.nl
            module.grid = [torch.zeros(1, device=device)] * module.nl
    with torch.no_grad():
        model = model.fuse().eval()
    nc = int(model.model[-1].nc)
    if nc != 1:
        raise ValueError(f"Expected single-class GLAD checkpoint, found nc={nc}: {path}")
    return model


class AppearanceDetector:
    def __init__(self, global_weights, local_weights, classifier_weights,
                 yolo_root, device="cpu", imgsz=640, threads=4,
                 global_conf=0.5, local_conf=0.1, nms_iou=0.4):
        if imgsz <= 0 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        if not (0 < global_conf <= 1 and 0 < local_conf <= 1 and 0 < nms_iou <= 1):
            raise ValueError("Confidence and NMS thresholds must be in (0, 1]")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        if threads > 0:
            torch.set_num_threads(threads)
        self.imgsz = int(imgsz)
        self.global_conf, self.local_conf = global_conf, local_conf
        self.nms_iou = nms_iou
        yolo_root = Path(yolo_root).resolve()
        if not (yolo_root / "models" / "yolo.py").is_file():
            raise FileNotFoundError("Missing YOLOv5 source; run python scripts/bootstrap.py")
        # Pickle uses upstream absolute module names. Refuse an already-imported
        # different YOLO tree instead of silently mixing incompatible code.
        for package in ("models", "utils"):
            existing = sys.modules.get(package)
            if existing is not None:
                location = str(getattr(existing, "__file__", "") or "")
                if location and not Path(location).resolve().is_relative_to(yolo_root):
                    raise RuntimeError(f"Another '{package}' package is loaded; start a fresh GLAD process")
        if str(yolo_root) not in sys.path:
            sys.path.insert(0, str(yolo_root))
        # v6.0 downloads a drawing font at import time on rank 0. This adapter
        # draws with OpenCV and does not use Annotator, so suppress that optional
        # side effect during import and keep its cache inside this project.
        previous_rank = os.environ.get("RANK")
        previous_config = os.environ.get("YOLOV5_CONFIG_DIR")
        cache_dir = yolo_root.parent.parent / ".cache" / "yolov5"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RANK"] = "1"
        os.environ["YOLOV5_CONFIG_DIR"] = str(cache_dir)
        try:
            importlib.import_module("models.yolo")
        finally:
            for name, previous in (("RANK", previous_rank), ("YOLOV5_CONFIG_DIR", previous_config)):
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
        for path in (global_weights, local_weights, classifier_weights):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing {path}; run python scripts/bootstrap.py")
        self.global_model = _load_yolo(global_weights, self.device)
        self.local_model = _load_yolo(local_weights, self.device)
        self.classifier = CandidateNet().to(self.device)
        state = torch.load(str(classifier_weights), map_location=self.device, weights_only=True)
        self.classifier.load_state_dict(state)
        self.classifier.eval()
        self.metadata = {"global_names": self.global_model.names,
                         "local_names": self.local_model.names,
                         "num_classes": 1, "network_size": self.imgsz,
                         "device": str(self.device)}

    @torch.inference_mode()
    def detect(self, image, local=False):
        tensor, scale, pad = letterbox(image, self.imgsz)
        tensor = tensor.to(self.device)
        model = self.local_model if local else self.global_model
        confidence = self.local_conf if local else self.global_conf
        prediction = model(tensor)[0][0]
        # Published GLAD weights contain a single MAV class. Confidence is the
        # product of objectness and class probability, as in YOLOv5 NMS.
        scores = prediction[:, 4] * prediction[:, 5]
        keep = torch.isfinite(prediction).all(dim=1) & (scores >= confidence)
        prediction, scores = prediction[keep], scores[keep]
        if not len(prediction):
            return []
        boxes = torch.cat((prediction[:, :2] - prediction[:, 2:4] / 2,
                           prediction[:, :2] + prediction[:, 2:4] / 2), dim=1)
        selected = nms(boxes, scores, self.nms_iou)[:100]
        boxes = restore_boxes(boxes[selected], scale, pad, image.shape).cpu().tolist()
        scores = scores[selected].cpu().tolist()
        source = "LAD" if local else "GAD"
        return [Detection(tuple(box), float(score), source)
                for box, score in zip(boxes, scores)
                if box[2] > box[0] and box[3] > box[1]]

    @torch.inference_mode()
    def classify(self, crop):
        if crop is None or not crop.size:
            return 0.0
        # The official classifier was inferred with BGR, not RGB.
        image = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        tensor = torch.from_numpy(image).float().div_(255).unsqueeze(0).to(self.device)
        return float(self.classifier(tensor).softmax(dim=1)[0, 1].item())
