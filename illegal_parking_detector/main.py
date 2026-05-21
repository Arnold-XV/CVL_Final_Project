import cv2
import os

from src.preprocessing import (
    PreprocessConfig,
    PreprocessingPipeline
)

FILENAME = "f5.mp4"
path = os.path.join("../cctv_recordings/", FILENAME)

def main():

    config = PreprocessConfig(
        VIDEO_SOURCE=path,
        TARGET_WIDTH=1280,
        TARGET_HEIGHT=720,
        FRAME_SKIP=2,
        ENABLE_CLAHE=False,
        ENABLE_GAMMA=False,
        SHOW_PREVIEW=True
    )

    pipeline = PreprocessingPipeline(config)

    try:
        for data in pipeline.process_stream():

            frame = data.processed_frame.copy()

            cv2.putText(
                frame,
                f"Frame: {data.frame_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Timestamp: {data.timestamp:.2f}s",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Preprocessing Output",
                frame
            )

            key = cv2.waitKey(1)

            if key == 27:
                break

    finally:
        pipeline.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()