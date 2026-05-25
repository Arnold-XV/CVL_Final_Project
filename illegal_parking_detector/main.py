import argparse

from src.config import load_config
from src.pipeline import IllegalParkingDetectionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Illegal Parking Detection using YOLO26 + Lucas-Kanade"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config\\config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override video source path/URL from config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.source:
        config.VIDEO_SOURCE = args.source

    pipeline = IllegalParkingDetectionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
