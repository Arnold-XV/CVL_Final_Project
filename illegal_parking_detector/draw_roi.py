import cv2
import json
import numpy as np

# Pastikan path video dan resolusi sama dengan di config.yaml
VIDEO_PATH = "../cctv_recordings/f5.mp4"
WIDTH, HEIGHT = 1280, 720 

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Titik ditambahkan: ({x}, {y})")

cap = cv2.VideoCapture(VIDEO_PATH)
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
        with open("roi.json", "w") as f:
            json.dump(points, f)
        print("Mantap! File roi.json berhasil disimpan.")
        break
    elif key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()