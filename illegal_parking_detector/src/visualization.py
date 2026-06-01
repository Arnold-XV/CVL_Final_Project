import cv2
import csv
import time
from datetime import datetime


class VisualizationOutputModule:
    def __init__(
        self,
        output_video_path="output_result.mp4",
        output_log_path="violation_log.csv",
        fps=30,
        frame_size=(1280, 720)
    ):

        # Video writer initialization
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.video_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            frame_size
        )

        # CSV log initialization
        self.log_file = open(output_log_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        # CSV header
        self.csv_writer.writerow([
            "timestamp",
            "vehicle_id",
            "status",
            "duration_stopped_sec"
        ])

        # Statistics
        self.total_violations = 0
        self.start_time = time.time()

    # Draw bounding boxes and vehicle status
    def draw_vehicle_info(
        self,
        frame,
        vehicle_id,
        bbox,
        status,
        stopped_duration=0
    ):

        x1, y1, x2, y2 = bbox

        # Color configuration
        if status == "MOVING":
            color = (0, 255, 0)      # Green

        elif status == "STOPPED":
            color = (0, 255, 255)    # Yellow

        elif status == "ILLEGAL PARKING":
            color = (0, 0, 255)      # Red

        else:
            color = (255, 255, 255)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Label text
        label = f"ID:{vehicle_id} | {status}"

        # Add stop duration if applicable
        if status != "MOVING":
            label += f" | {stopped_duration:.1f}s"

        # Draw text background
        cv2.rectangle(
            frame,
            (x1, y1 - 30),
            (x1 + 320, y1),
            color,
            -1
        )

        # Put label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        return frame

    # Draw system statistics
    def draw_statistics(
        self,
        frame,
        current_frame_number
    ):

        elapsed_time = time.time() - self.start_time

        fps = current_frame_number / elapsed_time \
            if elapsed_time > 0 else 0

        stats_text_1 = f"FPS: {fps:.2f}"
        stats_text_2 = f"Violations: {self.total_violations}"

        cv2.putText(
            frame,
            stats_text_1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            stats_text_2,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return frame

    # Log violation event
    def log_violation(
        self,
        vehicle_id,
        duration_stopped
    ):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.csv_writer.writerow([
            timestamp,
            vehicle_id,
            "ILLEGAL PARKING",
            round(duration_stopped, 2)
        ])

        self.total_violations += 1

    # Save processed frame to output video
    def write_frame(self, frame):
        self.video_writer.write(frame)

    # Release resources
    def release(self):
        self.video_writer.release()
        self.log_file.close()


