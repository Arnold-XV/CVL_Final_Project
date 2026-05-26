from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class DetectedViolation:
    vehicle_id: int
    start_time_sec: float
    end_time_sec: Optional[float] = None  # None if still ongoing at video end
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class GroundTruthViolation:
    start_time_sec: float
    end_time_sec: float
    bbox: Optional[Tuple[int, int, int, int]] = None

@dataclass
class ObjectDetectionEvaluator:
    # --- Per-frame runtime stats ---
    inference_times: List[float] = field(default_factory=list)
    detections_per_frame: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    # --- Tracking stats ---
    total_tracks_created: int = 0
    total_tracks_lost: int = 0

    # --- State distribution counters (frame-level) ---
    state_counts: Dict[str, int] = field(
        default_factory=lambda: {"MOVING": 0, "STOPPED": 0, "ILLEGAL PARKING": 0}
    )

    # --- Violation records ---
    detected_violations: List[DetectedViolation] = field(default_factory=list)
    _active_violations: Dict[int, DetectedViolation] = field(default_factory=dict)

    # --- Ground truth ---
    ground_truth_violations: List[GroundTruthViolation] = field(default_factory=list)

    # ── Per-frame update (called every frame) ─────────────────────
    def update(
        self,
        detection_count: int,
        inference_time: float,
        confidences: Optional[List[float]] = None,
    ) -> None:
        self.detections_per_frame.append(detection_count)
        self.inference_times.append(inference_time)
        if confidences:
            self.confidences.extend(confidences)

    # ── Track lifecycle ───────────────────────────────────────────
    def track_created(self) -> None:
        self.total_tracks_created += 1

    def track_lost(self) -> None:
        self.total_tracks_lost += 1

    # ── State distribution update ─────────────────────────────────
    def record_vehicle_state(self, state: str) -> None:
        if state in self.state_counts:
            self.state_counts[state] += 1

    # ── Violation lifecycle ───────────────────────────────────────
    def violation_started(
        self,
        vehicle_id: int,
        start_time: float,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        violation = DetectedViolation(
            vehicle_id=vehicle_id,
            start_time_sec=start_time,
            bbox=bbox,
        )
        self._active_violations[vehicle_id] = violation

    def violation_ended(self, vehicle_id: int, end_time: float) -> None:
        if vehicle_id in self._active_violations:
            violation = self._active_violations.pop(vehicle_id)
            violation.end_time_sec = end_time
            self.detected_violations.append(violation)

    def finalize_violations(self, final_timestamp: float) -> None:
        for vehicle_id in list(self._active_violations.keys()):
            violation = self._active_violations.pop(vehicle_id)
            violation.end_time_sec = final_timestamp
            self.detected_violations.append(violation)

    # ── Ground-truth loading ──────────────────────────────────────
    def load_ground_truth(self, gt_path: str) -> None:
        path = Path(gt_path)
        if not path.exists():
            print(f"[Evaluator] Ground truth file not found: {path}")
            return

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        violations = data.get("violations", [])
        for v in violations:
            gt = GroundTruthViolation(
                start_time_sec=float(v["start_time_sec"]),
                end_time_sec=float(v["end_time_sec"]),
                bbox=tuple(v["bbox"]) if v.get("bbox") else None,
            )
            self.ground_truth_violations.append(gt)

        print(
            f"[Evaluator] Loaded {len(self.ground_truth_violations)} "
            f"ground-truth violation(s) from {path.name}"
        )

    # ── Temporal IoU between two time intervals ───────────────────
    @staticmethod
    def _temporal_iou(
        start_a: float, end_a: float,
        start_b: float, end_b: float,
    ) -> float:
        intersection_start = max(start_a, start_b)
        intersection_end = min(end_a, end_b)
        intersection = max(0.0, intersection_end - intersection_start)

        union = (end_a - start_a) + (end_b - start_b) - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    # ── Match detections to ground truth ──────────────────────────
    def _match_violations(
        self, temporal_iou_threshold: float = 0.1
    ) -> Dict:
        """
        Greedy matching between detected and ground-truth violations
        based on temporal IoU.

        Returns dict with TP, FP, FN counts and per-match details.
        """
        if not self.ground_truth_violations:
            return {
                "tp": 0, "fp": len(self.detected_violations), "fn": 0,
                "matches": [], "note": "no_ground_truth",
            }

        # Build candidate pairs sorted by temporal IoU (descending)
        pairs = []
        for det_idx, det in enumerate(self.detected_violations):
            det_end = det.end_time_sec or det.start_time_sec
            for gt_idx, gt in enumerate(self.ground_truth_violations):
                iou = self._temporal_iou(
                    det.start_time_sec, det_end,
                    gt.start_time_sec, gt.end_time_sec,
                )
                if iou >= temporal_iou_threshold:
                    pairs.append((iou, det_idx, gt_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_det = set()
        matched_gt = set()
        matches = []

        for iou, det_idx, gt_idx in pairs:
            if det_idx in matched_det or gt_idx in matched_gt:
                continue
            matched_det.add(det_idx)
            matched_gt.add(gt_idx)
            matches.append({
                "det_idx": det_idx,
                "gt_idx": gt_idx,
                "temporal_iou": round(iou, 4),
                "det_vehicle_id": self.detected_violations[det_idx].vehicle_id,
            })

        tp = len(matches)
        fp = len(self.detected_violations) - tp
        fn = len(self.ground_truth_violations) - tp

        return {"tp": tp, "fp": fp, "fn": fn, "matches": matches}

    # ── Compute all metrics ───────────────────────────────────────
    def compute_metrics(self) -> dict:
        """
        Compute comprehensive evaluation metrics.

        Always returns runtime metrics. Adds violation-level precision/recall/F1
        and temporal IoU when ground truth is available.
        """
        # --- Runtime / detection metrics (always available) ---
        total_frames = len(self.inference_times)

        if total_frames > 0:
            avg_inference = sum(self.inference_times) / total_frames
            avg_fps = 1.0 / avg_inference if avg_inference > 0 else 0.0
            min_inference = min(self.inference_times)
            max_inference = max(self.inference_times)
        else:
            avg_inference = 0.0
            avg_fps = 0.0
            min_inference = 0.0
            max_inference = 0.0

        total_detections = sum(self.detections_per_frame)
        avg_detections = (
            total_detections / total_frames if total_frames > 0 else 0.0
        )

        # Confidence stats
        if self.confidences:
            conf_sorted = sorted(self.confidences)
            avg_confidence = sum(conf_sorted) / len(conf_sorted)
            median_confidence = conf_sorted[len(conf_sorted) // 2]
            min_confidence = conf_sorted[0]
            max_confidence = conf_sorted[-1]
        else:
            avg_confidence = None
            median_confidence = None
            min_confidence = None
            max_confidence = None

        # State distribution percentages
        total_state_obs = sum(self.state_counts.values())
        if total_state_obs > 0:
            state_distribution = {
                k: {
                    "count": v,
                    "percentage": round(v / total_state_obs * 100, 2),
                }
                for k, v in self.state_counts.items()
            }
        else:
            state_distribution = self.state_counts

        # Violation summary
        total_violations = len(self.detected_violations)
        violation_durations = []
        for v in self.detected_violations:
            if v.end_time_sec is not None:
                violation_durations.append(v.end_time_sec - v.start_time_sec)

        avg_violation_duration = (
            sum(violation_durations) / len(violation_durations)
            if violation_durations else 0.0
        )

        metrics = {
            # --- Performance ---
            "total_frames_processed": total_frames,
            "avg_fps": round(avg_fps, 2),
            "avg_inference_time_sec": round(avg_inference, 6),
            "min_inference_time_sec": round(min_inference, 6),
            "max_inference_time_sec": round(max_inference, 6),

            # --- Detection ---
            "total_detections": total_detections,
            "avg_detections_per_frame": round(avg_detections, 2),
            "confidence_stats": {
                "avg": round(avg_confidence, 4) if avg_confidence is not None else None,
                "median": round(median_confidence, 4) if median_confidence is not None else None,
                "min": round(min_confidence, 4) if min_confidence is not None else None,
                "max": round(max_confidence, 4) if max_confidence is not None else None,
                "total_samples": len(self.confidences),
            },

            # --- Tracking ---
            "tracking": {
                "total_tracks_created": self.total_tracks_created,
                "total_tracks_lost": self.total_tracks_lost,
                "track_fragmentation_ratio": (
                    round(self.total_tracks_lost / self.total_tracks_created, 4)
                    if self.total_tracks_created > 0 else 0.0
                ),
            },

            # --- State distribution ---
            "state_distribution": state_distribution,

            # --- Violations ---
            "violations": {
                "total_detected": total_violations,
                "avg_duration_sec": round(avg_violation_duration, 2),
                "durations": [round(d, 2) for d in violation_durations],
            },
        }

        # --- Ground-truth evaluation (only when GT is available) ---
        if self.ground_truth_violations:
            match_result = self._match_violations()
            tp = match_result["tp"]
            fp = match_result["fp"]
            fn = match_result["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            avg_temporal_iou = 0.0
            if match_result["matches"]:
                avg_temporal_iou = sum(
                    m["temporal_iou"] for m in match_result["matches"]
                ) / len(match_result["matches"])

            metrics["ground_truth_evaluation"] = {
                "ground_truth_violations": len(self.ground_truth_violations),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "avg_temporal_iou": round(avg_temporal_iou, 4),
                "match_details": match_result["matches"],
            }
        else:
            metrics["ground_truth_evaluation"] = {
                "status": "no_ground_truth_provided",
                "note": (
                    "Gunakan tool annotate_ground_truth.py untuk membuat "
                    "file anotasi, lalu set GROUND_TRUTH_PATH di config.yaml"
                ),
            }

        return metrics

def save_metrics_to_json(metrics: dict, output_path: Optional[str]) -> None:
    if not output_path:
        print("[Warning] OUTPUT_METRICS_PATH is empty; metrics not saved.")
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"[Evaluator] Metrics saved to: {output}")
