from __future__ import annotations
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    from .config import SystemConfig
except ImportError:
    from config import SystemConfig  # fallback untuk run langsung / testing

try:
    from .detector import Detection
except ImportError:
    from detector import Detection  # fallback


Point2D = Tuple[float, float]

TrackingResult = Tuple[Optional[Point2D], float, bool]

# CLASS: LucasKanadeTracker
class LucasKanadeTracker:

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

        self.lk_params: dict = dict(
            winSize=config.LK_WIN_SIZE,        
            maxLevel=config.LK_MAX_LEVEL,      
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.LK_TERM_CRITERIA_COUNT,  
                config.LK_TERM_CRITERIA_EPS,   
            ),
        )

        self.prev_gray: Optional[np.ndarray] = None

    def compute_displacements(
        self,
        curr_frame: np.ndarray,
        prev_detections: List[Detection],
    ) -> List[TrackingResult]:
        
        if self.prev_gray is None or len(prev_detections) == 0:
            return [(None, 0.0, False)] * len(prev_detections)

        prev_midpoints: List[Point2D] = [det.midpoint for det in prev_detections]

        curr_gray: np.ndarray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        p0: np.ndarray = np.array(prev_midpoints, dtype=np.float32).reshape(-1, 1, 2)

        p1, st, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,  
            curr_gray,       
            p0,              
            None,            
            **self.lk_params,
        )

        results: List[TrackingResult] = []

        for i, (old_x, old_y) in enumerate(prev_midpoints):
            tracking_ok: bool = (
                st is not None
                and p1 is not None
                and bool(st[i][0] == 1)
            )

            if tracking_ok:
                new_x = float(p1[i][0][0])
                new_y = float(p1[i][0][1])

                # Displacement magnitude = jarak Euclidean
                dx: float = new_x - old_x
                dy: float = new_y - old_y
                magnitude: float = float(np.sqrt(dx * dx + dy * dy))

                results.append(((new_x, new_y), magnitude, True))
            else:
                # Point loss: titik tidak dapat dilacak di frame ini
                results.append((None, 0.0, False))

        return results

    def update_prev_frame(self, curr_frame: np.ndarray) -> None:
       self.prev_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    def is_initialized(self) -> bool:
        return self.prev_gray is not None

    def reset(self) -> None:
        
        self.prev_gray = None


# CLASS: DisplacementSmoother
class DisplacementSmoother:
    def __init__(self, window_size: int, displacement_threshold: float) -> None:
        self.window_size: int = window_size
        self.displacement_threshold: float = displacement_threshold
        self._buffer: deque = deque(maxlen=window_size)

    def update(self, raw_displacement: float) -> float:
        self._buffer.append(raw_displacement)
        return self.get_smoothed()

    def get_smoothed(self) -> float:
        return float(np.mean(self._buffer)) if self._buffer else 0.0

    def is_stationary(self) -> bool:
        if len(self._buffer) < self.window_size:
            return False  # data belum cukup → anggap bergerak
        return self.get_smoothed() < self.displacement_threshold

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def buffer_fill_ratio(self) -> float:
        return len(self._buffer) / self.window_size if self.window_size > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"DisplacementSmoother("
            f"window={self.window_size}, "
            f"threshold={self.displacement_threshold}px, "
            f"smoothed={self.get_smoothed():.3f}px, "
            f"stationary={self.is_stationary()})"
        )


# CLASS: MultiVehicleSmoother
class MultiVehicleSmoother:
    def __init__(self, config: SystemConfig) -> None:
        self._window_size: int = config.SMOOTHING_WINDOW_SIZE
        self._threshold: float = config.DISPLACEMENT_THRESHOLD
        self._smoothers: Dict[int, DisplacementSmoother] = {}

    def register(self, vehicle_id: int) -> None:
        if vehicle_id not in self._smoothers:
            self._smoothers[vehicle_id] = DisplacementSmoother(
                window_size=self._window_size,
                displacement_threshold=self._threshold,
            )

    def update(self, vehicle_id: int, raw_displacement: float) -> float:
        self.register(vehicle_id)
        return self._smoothers[vehicle_id].update(raw_displacement)

    def get_smoothed(self, vehicle_id: int) -> float:
        smoother = self._smoothers.get(vehicle_id)
        return smoother.get_smoothed() if smoother else 0.0

    def is_stationary(self, vehicle_id: int) -> bool:
        smoother = self._smoothers.get(vehicle_id)
        return smoother.is_stationary() if smoother else False

    def remove(self, vehicle_id: int) -> None:
        self._smoothers.pop(vehicle_id, None)

    def reset_vehicle(self, vehicle_id: int) -> None:
        smoother = self._smoothers.get(vehicle_id)
        if smoother:
            smoother.reset()

    def active_ids(self) -> List[int]:
        return list(self._smoothers.keys())

    def summary(self) -> Dict[int, dict]:
        return {
            vid: {
                "smoothed_displacement": s.get_smoothed(),
                "is_stationary": s.is_stationary(),
                "buffer_fill_ratio": s.buffer_fill_ratio,
            }
            for vid, s in self._smoothers.items()
        }

    def __repr__(self) -> str:
        return (
            f"MultiVehicleSmoother("
            f"vehicles={len(self._smoothers)}, "
            f"window={self._window_size}, "
            f"threshold={self._threshold}px)"
        )


# UTILITY FUNCTIONS
def compute_euclidean_displacement(
    point_a: Point2D,
    point_b: Point2D,
) -> float:
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return float(np.sqrt(dx * dx + dy * dy))


def apply_moving_average(values: List[float], window_size: int) -> List[float]:
    if window_size < 1:
        raise ValueError(f"window_size harus >= 1, diterima: {window_size}")
    if not values:
        return []

    smoothed: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        smoothed.append(float(np.mean(values[start: i + 1])))
    return smoothed