import argparse
import cv2
import json
import numpy as np
import os

WIDTH, HEIGHT = 1280, 720 

def get_args():
    parser = argparse.ArgumentParser(description="Gambar ROI untuk video CCTV")
    parser.add_argument("--video", type=str, required=True, help="Path ke file video CCTV")
    return parser.parse_args()

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Titik ditambahkan: ({x}, {y})")

args = get_args()
video_path = args.video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Gagal membuka video!")
    exit()

frame = cv2.resize(frame, (WIDTH, HEIGHT))
cv2.namedWindow("Gambar ROI - Klik Kiri Tambah Titik, Tekan 'S' untuk Save")
cv2.setMouseCallback("Gambar ROI - Klik Kiri Tambah Titik, Tekan 'S' untuk Save", mouse_callback)

print("Instruksi: Klik kiri pada gambar untuk membentuk area larangan parkir (hindari jalan utama/lampu merah).")
print("Jika sudah tertutup/selesai, tekan tombol 'S' di keyboard untuk menyimpan.")

while True:
    temp_frame = frame.copy()
    
    if len(points) > 0:
        cv2.polylines(temp_frame, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=2)
        for p in points:
            cv2.circle(temp_frame, tuple(p), 5, (0, 255, 0), -1)
            
    cv2.imshow("Gambar ROI - Klik Kiri Tambah Titik, Tekan 'S' untuk Save", temp_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_roi_file = f"roi_{video_stem}.json"
        with open(output_roi_file, "w") as f:
            json.dump(points, f)
        print(f"Mantap! File {output_roi_file} berhasil disimpan.")
        break
    elif key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()