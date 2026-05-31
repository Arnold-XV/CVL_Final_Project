import argparse
import copy
from pathlib import Path

from src.config import load_config
from src.pipeline import IllegalParkingDetectionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Illegal Parking Detection using YOLO26 + Lucas-Kanade"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override video source path/URL from config.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Path ke folder yang berisi banyak video CCTV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.source_dir:
        source_dir_path = Path(args.source_dir)
        video_files = list(source_dir_path.glob("*.mp4"))
        
        for video_file in video_files:
            print(f"\n[{video_file.name}] Memulai pemrosesan...")
            
            current_config = copy.deepcopy(config)
            current_config.VIDEO_SOURCE = str(video_file)
            
            video_stem = video_file.stem
            
            vid_out_path = Path(current_config.OUTPUT_VIDEO_PATH)
            current_config.OUTPUT_VIDEO_PATH = str(vid_out_path.with_name(f"{vid_out_path.stem}_{video_stem}{vid_out_path.suffix}"))
            
            log_out_path = Path(current_config.OUTPUT_LOG_PATH)
            current_config.OUTPUT_LOG_PATH = str(log_out_path.with_name(f"{log_out_path.stem}_{video_stem}{log_out_path.suffix}"))
            
            metrics_out_path = Path(current_config.OUTPUT_METRICS_PATH)
            current_config.OUTPUT_METRICS_PATH = str(metrics_out_path.with_name(f"{metrics_out_path.stem}_{video_stem}{metrics_out_path.suffix}"))
            
            roi_path = Path(current_config.ROI_DEFINITION_PATH).parent / f"roi_{video_stem}.json"
            if roi_path.exists():
                current_config.ROI_DEFINITION_PATH = str(roi_path)
                print(f"[{video_file.name}] Menggunakan ROI: {roi_path.name}")
            else:
                current_config.ROI_DEFINITION_PATH = "" 
                print(f"[{video_file.name}] Peringatan: File {roi_path.name} tidak ditemukan. ROI dinonaktifkan.")
            
            pipeline = IllegalParkingDetectionPipeline(current_config)
            pipeline.run()
    else:
        if args.source:
            config.VIDEO_SOURCE = args.source

        pipeline = IllegalParkingDetectionPipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
