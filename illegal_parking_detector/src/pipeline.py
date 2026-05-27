from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import SystemConfig
from .decision_logic import VehicleState, VehicleStateManager
from .detector import Detection, VehicleDetector
from .evaluator import ObjectDetectionEvaluator, save_metrics_to_json
from .preprocessing import PreprocessConfig, PreprocessingPipeline
from .tracker import LucasKanadeTracker, MultiVehicleSmoother
from .visualization import VisualizationOutputModule


Point2D = Tuple[float, float]


class IllegalParkingDetectionPipeline:
    def __init__(self, config: SystemConfig):
        self.config = config

        self.preprocessor = PreprocessingPipeline(
            PreprocessConfig(
                VIDEO_SOURCE=config.VIDEO_SOURCE,
                TARGET_WIDTH=config.INPUT_WIDTH,
                TARGET_HEIGHT=config.INPUT_HEIGHT,
                FRAME_SKIP=config.FRAME_SKIP,
                ENABLE_CLAHE=config.ENABLE_CLAHE,
                ENABLE_GAMMA=config.ENABLE_GAMMA,
                GAMMA_VALUE=config.GAMMA_VALUE,
                SHOW_PREVIEW=False,
            )
        )

        self.detector = VehicleDetector(config)
        self.tracker = LucasKanadeTracker(config)
        self.smoother = MultiVehicleSmoother(config)
        self.state_manager = VehicleStateManager(
            illegal_parking_threshold_sec=config.PARKING_DURATION_THRESHOLD,
            log_file_path=config.OUTPUT_LOG_PATH,
        )
        self.evaluator = ObjectDetectionEvaluator()

        # Load ground-truth annotations if available
        if config.GROUND_TRUTH_PATH and config.GROUND_TRUTH_PATH.strip():
            self.evaluator.load_ground_truth(config.GROUND_TRUTH_PATH)

        self._tracks: Dict[int, Detection] = {}
        self._missing_counts: Dict[int, int] = {}
        self._next_track_id = 0
        self._visual_logged: set[int] = set()
        self._last_timestamp: float = 0.0

        self.roi_polygon = self._load_roi_polygon(config.ROI_DEFINITION_PATH)
        self.visual_log_path = self._derive_visual_log_path(config.OUTPUT_LOG_PATH)
        self._ensure_output_dirs()

        fps = self.preprocessor.metadata.get("fps") or config.VIDEO_FPS
        self.visualizer = VisualizationOutputModule(
            output_video_path=config.OUTPUT_VIDEO_PATH,
            output_log_path=self.visual_log_path,
            fps=fps,
            frame_size=(config.INPUT_WIDTH, config.INPUT_HEIGHT),
        )

    def _ensure_output_dirs(self) -> None:
        for path in [
            self.config.OUTPUT_VIDEO_PATH,
            self.config.OUTPUT_LOG_PATH,
            self.config.OUTPUT_METRICS_PATH,
            self.visual_log_path,
        ]:
            if not path:
                continue
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _derive_visual_log_path(self, output_log_path: str) -> str:
        if not output_log_path:
            return "outputs/logs/violations_visual.csv"
        path = Path(output_log_path)
        return str(path.with_name(f"{path.stem}_visual{path.suffix}"))

    def _load_roi_polygon(self, roi_path: str) -> Optional[np.ndarray]:
        if not roi_path:
            return None
        path = Path(roi_path)
        if not path.exists():
            print(f"[Warning] ROI definition not found: {path}. ROI disabled.")
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        points = None
        if isinstance(data, list):
            points = data
        elif isinstance(data, dict):
            for key in ("points", "roi", "polygon", "roi_polygon"):
                if key in data:
                    points = data[key]
                    break
        if not points:
            print(f"[Warning] ROI file has no valid points: {path}. ROI disabled.")
            return None

        polygon = np.array(points, dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            print(f"[Warning] ROI points must be Nx2, got shape {polygon.shape}. ROI disabled.")
            return None
        return polygon.reshape((-1, 1, 2))

    def _is_in_roi(self, point: Point2D) -> bool:
        if self.roi_polygon is None:
            return True
        return cv2.pointPolygonTest(self.roi_polygon, point, False) >= 0

    def _associate_detections(
        self,
        detections: List[Detection],
        previous_tracks: Dict[int, Detection],
    ) -> Dict[int, int]:
        if not detections or not previous_tracks:
            return {}

        pairs = []
        for det_index, det in enumerate(detections):
            for track_id, prev_det in previous_tracks.items():
                dx = det.midpoint[0] - prev_det.midpoint[0]
                dy = det.midpoint[1] - prev_det.midpoint[1]
                dist = float(np.sqrt(dx * dx + dy * dy))
                pairs.append((dist, track_id, det_index))

        pairs.sort(key=lambda item: item[0])
        assignments: Dict[int, int] = {}
        used_tracks = set()
        used_dets = set()

        for dist, track_id, det_index in pairs:
            if dist > self.config.MAX_ASSOCIATION_DISTANCE:
                continue
            if track_id in used_tracks or det_index in used_dets:
                continue
            assignments[det_index] = track_id
            used_tracks.add(track_id)
            used_dets.add(det_index)

        return assignments

    def _cleanup_missing_tracks(self) -> None:
        for track_id in list(self._missing_counts.keys()):
            if self._missing_counts[track_id] <= self.config.MAX_MISSING_FRAMES:
                continue
            self._missing_counts.pop(track_id, None)
            self._tracks.pop(track_id, None)
            self.smoother.remove(track_id)
            self.state_manager.records.pop(track_id, None)
            self._visual_logged.discard(track_id)
            self.evaluator.track_lost()
            self.evaluator.violation_ended(track_id, self._last_timestamp)

    def run(self) -> None:
        frame_count = 0

        try:
            for data in self.preprocessor.process_stream():
                frame_count += 1
                raw_frame = data.processed_frame.copy()

                infer_start = time.perf_counter()
                detections = self.detector.detect(raw_frame)
                infer_time = time.perf_counter() - infer_start
                det_confidences = [det.confidence for det in detections]
                self.evaluator.update(len(detections), infer_time, det_confidences)

                detections = [
                    det for det in detections if self._is_in_roi(det.midpoint)
                ]

                prev_track_ids = list(self._tracks.keys())
                prev_detections = [self._tracks[tid] for tid in prev_track_ids]
                displacements = self.tracker.compute_displacements(raw_frame, prev_detections)
                displacement_by_track = {
                    track_id: (magnitude, ok)
                    for track_id, (_point, magnitude, ok) in zip(prev_track_ids, displacements)
                }

                assignments = self._associate_detections(detections, self._tracks)
                for det_index in range(len(detections)):
                    if det_index not in assignments:
                        assignments[det_index] = self._next_track_id
                        self._missing_counts[self._next_track_id] = 0
                        self._next_track_id += 1
                        self.evaluator.track_created()

                matched_track_ids = set(assignments.values())
                for track_id in list(self._tracks.keys()):
                    if track_id in matched_track_ids:
                        self._missing_counts[track_id] = 0
                    else:
                        self._missing_counts[track_id] = self._missing_counts.get(track_id, 0) + 1

                self._cleanup_missing_tracks()

                for det_index, det in enumerate(detections):
                    track_id = assignments[det_index]
                    self._tracks[track_id] = det

                annotated_frame = raw_frame.copy()

                for det_index, det in enumerate(detections):
                    track_id = assignments[det_index]
                    in_roi = self._is_in_roi(det.midpoint)
                    magnitude, ok = displacement_by_track.get(track_id, (0.0, False))

                    if ok:
                        self.smoother.update(track_id, magnitude)
                        is_stationary = in_roi and self.smoother.is_stationary(track_id)
                    else:
                        self.smoother.reset_vehicle(track_id)
                        is_stationary = False

                    record = self.state_manager.update_vehicle(
                        vehicle_id=track_id,
                        is_stationary=is_stationary,
                        timestamp=data.timestamp,
                        bbox=det.bbox,
                    )

                    annotated_frame = self.visualizer.draw_vehicle_info(
                        frame=annotated_frame,
                        vehicle_id=track_id,
                        bbox=det.bbox,
                        status=record.state,
                        stopped_duration=record.stopped_duration,
                    )

                    # Record state for evaluator statistics
                    self.evaluator.record_vehicle_state(record.state)

                    if (
                        record.state == VehicleState.ILLEGAL_PARKING
                        and track_id not in self._visual_logged
                    ):
                        self.visualizer.log_violation(track_id, record.stopped_duration)
                        self.evaluator.violation_started(
                            vehicle_id=track_id,
                            start_time=record.stop_start_time or data.timestamp,
                            bbox=det.bbox,
                        )
                        self._visual_logged.add(track_id)

                if self.roi_polygon is not None:
                    cv2.polylines(
                        annotated_frame,
                        [self.roi_polygon],
                        isClosed=True,
                        color=(255, 0, 255),
                        thickness=2,
                    )

                annotated_frame = self.visualizer.draw_statistics(annotated_frame, frame_count)
                self.visualizer.write_frame(annotated_frame)

                if self.config.SHOW_PREVIEW:
                    cv2.imshow("Illegal Parking Detection", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                self.tracker.update_prev_frame(raw_frame)
                self._last_timestamp = data.timestamp

        finally:
            self.evaluator.finalize_violations(self._last_timestamp)
            metrics = self.evaluator.compute_metrics()
            save_metrics_to_json(metrics, self.config.OUTPUT_METRICS_PATH)

            self.preprocessor.release()
            self.visualizer.release()
            cv2.destroyAllWindows()
