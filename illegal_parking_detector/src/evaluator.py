from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ObjectDetectionEvaluator:
    inference_times: List[float] = field(default_factory=list)
    detections_per_frame: List[int] = field(default_factory=list)

    def update(self, detection_count: int, inference_time: float) -> None:
        self.detections_per_frame.append(detection_count)
        self.inference_times.append(inference_time)

    def compute_metrics(self) -> dict:
        if not self.inference_times:
            avg_inference = 0.0
            avg_fps = 0.0
        else:
            avg_inference = sum(self.inference_times) / len(self.inference_times)
            avg_fps = 1.0 / avg_inference if avg_inference > 0 else 0.0

        total_detections = sum(self.detections_per_frame)

        return {
            "avg_fps": avg_fps,
            "avg_inference_time": avg_inference,
            "total_detections": total_detections,
            "precision": None,
            "recall": None,
            "map_at_0_5": None,
        }


def save_metrics_to_json(metrics: dict, output_path: Optional[str]) -> None:
    if not output_path:
        print("[Warning] OUTPUT_METRICS_PATH is empty; metrics not saved.")
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Evaluator] Metrics saved to: {output}")
