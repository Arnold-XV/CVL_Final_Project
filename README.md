# Illegal Parking Detection System

## Overview
This repository contains the final project for the Computer Vision course (Group 4). It implements an automated Illegal Parking Detection system using CCTV footage. The system integrates YOLO-based object detection with Lucas-Kanade optical flow tracking to identify and monitor vehicles, determining if they have remained stationary in restricted zones beyond an allowable time threshold.

## Contoh Hasil
> ![gif - Made with Clipchamp (1)](https://github.com/user-attachments/assets/ef024851-5303-4d50-99ec-421a1fea66fa)

## Key Features
- **Vehicle Detection:** Utilizes YOLO architecture (via Ultralytics) to accurately detect various classes of vehicles including cars, trucks, motorcycles, and buses.
- **Object Tracking & Displacement Analysis:** Employs the Lucas-Kanade optical flow algorithm to track vehicle movement across frames and calculate displacement.
- **Region of Interest (ROI) Configuration:** Includes interactive tools to define specific restricted parking zones within the camera's field of view.
- **Configurable Pipeline:** Highly customizable detection confidence, tracking thresholds, and parking duration limits via a centralized YAML configuration.
- **Ground Truth Annotation Tool:** Built-in tool for annotating video datasets to evaluate and validate system performance.
- **Batch Processing:** Supports processing multiple video streams sequentially.

## Repository Structure
- `illegal_parking_detector/`: The core application directory.
  - `main.py`: Entry point for running the detection pipeline.
  - `src/`: Contains source code modules including the detector, tracker, decision logic, and evaluation scripts.
  - `config/`: Holds `config.yaml` for system tuning.
  - `models/`: Directory for storing YOLO model weights (e.g., `yolo26s.pt`).
  - `draw_roi.py`: Utility script for defining ROI polygons on video frames.
  - `annotate_ground_truth.py`: Utility script for manual annotation of illegal parking events.
  - `requirements.txt`: Python package dependencies.
- `cctv_recordings/`: Contains sample CCTV video recordings used for testing and evaluation.
- `Laporan Final Project Kelompok 4.pdf`: The final project report containing methodology and results.

## Prerequisites
- Python 3.8+
- Required Python packages (listed in `requirements.txt`):
  - `opencv-python>=4.9.0`
  - `ultralytics>=8.0.0`
  - `torch>=2.2.0`
  - `torchvision>=0.17.0`
  - `numpy>=1.26.0`
  - `pyyaml>=6.0`
  - `pandas>=2.0.0`
  - `scipy>=1.12.0`
  - `tqdm>=4.66.0`

## Installation
1. Clone this repository to your local machine.
2. Navigate to the project directory:
   ```bash
   cd CVL_Final_Project/illegal_parking_detector
   ```
3. Install the required dependencies (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Defining Region of Interest (ROI)
Before running the detection pipeline, define the restricted parking zones for your specific CCTV angle:
```bash
python draw_roi.py --video ../cctv_recordings/f1.mp4
```
- Left-click to add points outlining the restricted area.
- Press `S` to save the ROI configuration to a JSON file.

### 2. Running the Detection Pipeline
Execute the main pipeline using the predefined configuration:
```bash
python main.py --config config/config.yaml
```
To process an entire directory of videos:
```bash
python main.py --config config/config.yaml --source_dir ../cctv_recordings/
```

### 3. Annotating Ground Truth (For Evaluation)
To evaluate the model's accuracy, you can create ground truth data for a video:
```bash
python annotate_ground_truth.py --video ../cctv_recordings/f1.mp4
```
- `S`: Start marking an illegal parking violation.
- `E`: End marking the current violation.
- `W`: Save annotations.

## Configuration
System parameters can be tuned in `illegal_parking_detector/config/config.yaml`. Key parameters include:
- `DETECTION_CONFIDENCE_THRESHOLD`: Minimum confidence score for YOLO detections.
- `PARKING_DURATION_THRESHOLD`: Time in seconds a vehicle must remain stationary to be flagged.
- `DISPLACEMENT_THRESHOLD`: Maximum pixel movement allowed for a vehicle to still be considered stationary.
- `ROI_DEFINITION_PATH`: Path to the JSON file containing ROI coordinates.
