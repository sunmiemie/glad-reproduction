"""Shared types. All boxes use (x1, y1, x2, y2), exclusive upper bounds."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    box: Box
    confidence: float
    source: str

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def translated(self, dx: float, dy: float):
        a, b, c, d = self.box
        return Detection((a + dx, b + dy, c + dx, d + dy), self.confidence, self.source)


@dataclass
class FrameResult:
    index: int
    mode: str
    detection: Optional[Detection]
    roi: Optional[Tuple[int, int, int, int]]
    prediction: Optional[Tuple[float, float]]
    lost_frames: int
    elapsed_ms: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
