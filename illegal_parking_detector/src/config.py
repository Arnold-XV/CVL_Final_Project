from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class SystemConfig:
    # --- Input/Output ---
    VIDEO_SOURCE: str = "data/raw/input.mp4"
    OUTPUT_VIDEO_PATH: str = "outputs/videos/annotated_output.mp4"
    OUTPUT_LOG_PATH: str = "outputs/logs/violations_log.csv"
    OUTPUT_METRICS_PATH: str = "outputs/metrics/evaluation_results.json"

    # --- Preprocessing ---
    INPUT_WIDTH: int = 1280
    INPUT_HEIGHT: int = 720
    FRAME_SKIP: int = 2
    ENABLE_CLAHE: bool = False
    ENABLE_GAMMA: bool = False
    GAMMA_VALUE: float = 1.2
    SHOW_PREVIEW: bool = True
    ROI_DEFINITION_PATH: str = ""

    # --- YOLO26 Detection ---
    MODEL_WEIGHTS_PATH: str = "models/yolo26/weights/yolo26s.pt"
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.45
    VEHICLE_CLASSES: List[str] = field(
        default_factory=lambda: ["car", "truck", "motorcycle", "bus"]
    )

    # --- Lucas-Kanade Tracking ---
    LK_WIN_SIZE: Tuple[int, int] = (3, 3)
    LK_MAX_LEVEL: int = 3
    LK_TERM_CRITERIA_EPS: float = 0.03
    LK_TERM_CRITERIA_COUNT: int = 10
    MAX_ASSOCIATION_DISTANCE: float = 80.0

    # --- State Management ---
    DISPLACEMENT_THRESHOLD: float = 2.0
    SMOOTHING_WINDOW_SIZE: int = 5
    PARKING_DURATION_THRESHOLD: float = 30.0
    MAX_MISSING_FRAMES: int = 15

    # --- Video Processing ---
    VIDEO_FPS: float = 30.0


def _resolve_path(base_dir: Path, value: str) -> str:
    if value is None:
        return value
    if isinstance(value, str) and value.strip() == "":
        return value
    if isinstance(value, str) and "://" in value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(config_path: str = "config\\config.yaml") -> SystemConfig:
    config_file = Path(config_path)
    base_dir = config_file.resolve().parent
    config_data = {}

    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        if not isinstance(raw_data, dict):
            print(f"[Warning] Config file is not a mapping: {config_file}")
        else:
            config_data = raw_data
    else:
        print(f"[Warning] Config file not found: {config_file}. Using defaults.")

    valid_fields = set(SystemConfig.__dataclass_fields__.keys())
    unknown_keys = [key for key in config_data.keys() if key not in valid_fields]
    if unknown_keys:
        print(f"[Warning] Unknown config keys ignored: {unknown_keys}")

    filtered_data = {key: config_data[key] for key in valid_fields if key in config_data}
    config = SystemConfig(**filtered_data)

    config.VIDEO_SOURCE = _resolve_path(base_dir, config.VIDEO_SOURCE)
    config.OUTPUT_VIDEO_PATH = _resolve_path(base_dir, config.OUTPUT_VIDEO_PATH)
    config.OUTPUT_LOG_PATH = _resolve_path(base_dir, config.OUTPUT_LOG_PATH)
    config.OUTPUT_METRICS_PATH = _resolve_path(base_dir, config.OUTPUT_METRICS_PATH)
    config.ROI_DEFINITION_PATH = _resolve_path(base_dir, config.ROI_DEFINITION_PATH)
    config.MODEL_WEIGHTS_PATH = _resolve_path(base_dir, config.MODEL_WEIGHTS_PATH)

    return config
