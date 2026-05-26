import cv2
import time
from dataclasses import dataclass
from typing import Dict, Generator, Optional

# CONFIGURATION
@dataclass
class PreprocessConfig:
    VIDEO_SOURCE: str = "input.mp4"

    # Target resolution
    TARGET_WIDTH: int = 1280
    TARGET_HEIGHT: int = 720

    # Process every N-th frame
    FRAME_SKIP: int = 2

    # Optional enhancement
    ENABLE_CLAHE: bool = False
    ENABLE_GAMMA: bool = False
    GAMMA_VALUE: float = 1.2

    # Visualization
    SHOW_PREVIEW: bool = True


# VIDEO READER MODULE
class VideoReader:
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_metadata(self) -> Dict:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
        }

    def read_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


# RESIZE MODULE
class FrameResizer:
    def __init__(self, target_width: int, target_height: int):
        self.target_width = target_width
        self.target_height = target_height

    def resize(self, frame):
        """
        Resize while preserving aspect ratio using letterboxing.
        """

        h, w = frame.shape[:2]

        scale = min(
            self.target_width / w,
            self.target_height / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        canvas = self.create_letterbox_canvas(resized)

        return canvas

    def create_letterbox_canvas(self, resized_frame):
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        h, w = resized_frame.shape[:2]

        x_offset = (self.target_width - w) // 2
        y_offset = (self.target_height - h) // 2

        canvas[
            y_offset:y_offset + h,
            x_offset:x_offset + w
        ] = resized_frame

        return canvas


# OPTIONAL ENHANCEMENT MODULE
class FrameEnhancer:
    def __init__(self, enable_clahe=False, enable_gamma=False, gamma=1.2):
        self.enable_clahe = enable_clahe
        self.enable_gamma = enable_gamma
        self.gamma = gamma

    def apply(self, frame):
        if self.enable_clahe:
            frame = self.apply_clahe(frame)

        if self.enable_gamma:
            frame = self.apply_gamma(frame)

        return frame

    def apply_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, frame):
    inv_gamma = 1.0 / self.gamma
        
        table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        table = cv2.UMat(cv2.convertScaleAbs(cv2.merge([cv2.UMat(table).get().astype('uint8')] * 3))).get()
        
        return cv2.LUT(frame, table[:, :, 0])


# FRAME PACKAGING MODULE
@dataclass
class ProcessedFrame:
    frame_id: int
    timestamp: float
    original_frame: any
    processed_frame: any
    fps: float


# PREPROCESSING PIPELINE
class PreprocessingPipeline:
    def __init__(self, config: PreprocessConfig):
        self.config = config

        self.reader = VideoReader(config.VIDEO_SOURCE)

        self.resizer = FrameResizer(
            config.TARGET_WIDTH,
            config.TARGET_HEIGHT
        )

        self.enhancer = FrameEnhancer(
            config.ENABLE_CLAHE,
            config.ENABLE_GAMMA,
            config.GAMMA_VALUE
        )

        self.metadata = self.reader.get_metadata()

        print("\n========== VIDEO METADATA ==========")
        for key, value in self.metadata.items():
            print(f"{key}: {value}")
        print("====================================\n")

    def process_stream(self) -> Generator[ProcessedFrame, None, None]:
        frame_id = 0

        while True:
            ret, frame = self.reader.read_frame()

            if not ret:
                break

            # Frame skipping
            if frame_id % self.config.FRAME_SKIP != 0:
                frame_id += 1
                continue

            timestamp = frame_id / self.metadata["fps"]

            original_frame = frame.copy()

            # Resize
            processed_frame = self.resizer.resize(frame)

            # Optional enhancement
            processed_frame = self.enhancer.apply(processed_frame)

            packaged_frame = ProcessedFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                original_frame=original_frame,
                processed_frame=processed_frame,
                fps=self.metadata["fps"]
            )

            yield packaged_frame

            frame_id += 1

    def release(self):
        self.reader.release()


"""
Example usage from main.py:

from src.preprocessing.preprocessing_pipeline import (
    PreprocessConfig,
    PreprocessingPipeline
)

config = PreprocessConfig(
    VIDEO_SOURCE="data/raw/input.mp4",
    TARGET_WIDTH=1280,
    TARGET_HEIGHT=720,
    FRAME_SKIP=2
)

pipeline = PreprocessingPipeline(config)

for data in pipeline.process_stream():
    frame = data.processed_frame

    # Send frame to YOLO detector
    # detector.detect(frame)
"""

