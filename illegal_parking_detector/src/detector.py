"""YOLO26 detector wrapper using Ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from .config import SystemConfig


class Detection:
    """Structured detection result for a vehicle."""

    def __init__(
        self,
        bbox: tuple,
        confidence: float,
        class_label: str,
        class_id: int,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.class_label = class_label
        self.class_id = class_id
        self.midpoint = (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )

    def __repr__(self) -> str:
        return (
            f"Detection(class={self.class_label}, "
            f"conf={self.confidence:.2f}, "
            f"midpoint={self.midpoint})"
        )


class VehicleDetector:
    """
    Wrapper for YOLO26 model. Handles model loading, inference,
    and filtering outputs by class/confidence.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = self._load_model()
        self.class_names = self.model.names
        self.target_class_ids = {
            k
            for k, v in self.class_names.items()
            if v.lower() in [c.lower() for c in config.VEHICLE_CLASSES]
        }

    def _load_model(self):
        from ultralytics import YOLO

        weights = Path(self.config.MODEL_WEIGHTS_PATH)
        if not weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights}. Place weights at that path."
            )

        model = YOLO(str(weights))
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []

        results = self.model.predict(
            source=frame,
            conf=self.config.DETECTION_CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id not in self.target_class_ids:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf.item())
                class_label = self.class_names[class_id]
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_label=class_label,
                        class_id=class_id,
                    )
                )

        return detections
