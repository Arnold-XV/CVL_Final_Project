from __future__ import annotations
import os
import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple


# 1. VEHICLE STATES CONSTANTS
class VehicleState:
    MOVING = "MOVING"
    STOPPED = "STOPPED"
    ILLEGAL_PARKING = "ILLEGAL PARKING"


# 2. VEHICLE STATE RECORD
@dataclass
class VehicleStateRecord:
    vehicle_id: int
    state: str = VehicleState.MOVING
    stop_start_time: Optional[float] = None
    stopped_duration: float = 0.0
    violation_logged: bool = False
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_updated: float = 0.0

    def update_stopped_duration(self, current_time: float) -> float:
        if self.stop_start_time is not None:
            self.stopped_duration = current_time - self.stop_start_time
        return self.stopped_duration

    def transition_to(self, new_state: str, timestamp: float) -> None:
        if self.state == new_state:
            return

        # Transisi dari MOVING ke STOPPED
        if new_state == VehicleState.STOPPED:
            self.stop_start_time = timestamp
            self.stopped_duration = 0.0
            self.violation_logged = False
        
        # Transisi ke MOVING (kembali bergerak / jalan)
        elif new_state == VehicleState.MOVING:
            self.stop_start_time = None
            self.stopped_duration = 0.0
            self.violation_logged = False

        self.state = new_state


# 3. VEHICLE STATE MANAGER & DECISION LOGIC
class VehicleStateManager:
    def __init__(
        self,
        illegal_parking_threshold_sec: float = 10.0,
        log_file_path: str = "violation_events.csv",
        on_violation_callback: Optional[Callable[[int, float, Optional[Tuple[int, int, int, int]]], None]] = None
    ) -> None:
        self.illegal_parking_threshold_sec = illegal_parking_threshold_sec
        self.log_file_path = log_file_path
        self.on_violation_callback = on_violation_callback

        # Dictionary untuk menyimpan rekam jejak status per vehicle_id
        self.records: Dict[int, VehicleStateRecord] = {}

        # Inisialisasi file CSV log jika belum ada
        self._initialize_csv_log()

    def _initialize_csv_log(self) -> None:
        if not os.path.exists(self.log_file_path):
            try:
                # Membuat folder induk jika belum ada
                parent_dir = os.path.dirname(self.log_file_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

                with open(self.log_file_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "vehicle_id",
                        "status",
                        "duration_stopped_sec",
                        "bbox_x1_y1_x2_y2"
                    ])
            except Exception as e:
                print(f"[Warning] Gagal menginisialisasi file log {self.log_file_path}: {e}")

    def update_vehicle(
        self,
        vehicle_id: int,
        is_stationary: bool,
        timestamp: float,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> VehicleStateRecord:
        # Register kendaraan baru jika belum terdaftar
        if vehicle_id not in self.records:
            self.records[vehicle_id] = VehicleStateRecord(
                vehicle_id=vehicle_id,
                last_updated=timestamp,
                last_bbox=bbox
            )

        record = self.records[vehicle_id]
        record.last_updated = timestamp
        if bbox is not None:
            record.last_bbox = bbox

        # --- LOGIKA TRANSISI STATUS ---
        if is_stationary:
            # Jika kendaraan diam dan sebelumnya MOVING, ubah menjadi STOPPED
            if record.state == VehicleState.MOVING:
                record.transition_to(VehicleState.STOPPED, timestamp)
            
            # Jika dalam status STOPPED atau ILLEGAL PARKING, perbarui durasi berhenti
            elif record.state in [VehicleState.STOPPED, VehicleState.ILLEGAL_PARKING]:
                record.update_stopped_duration(timestamp)

                # Cek aturan batas waktu untuk menentukan parkir liar
                if (
                    record.state == VehicleState.STOPPED 
                    and record.stopped_duration >= self.illegal_parking_threshold_sec
                ):
                    record.transition_to(VehicleState.ILLEGAL_PARKING, timestamp)
                    # Trigger logging kejadian pelanggaran (hanya sekali per kejadian)
                    self._log_violation_event(record)
        else:
            # Jika kendaraan bergerak kembali, ubah status menjadi MOVING
            if record.state in [VehicleState.STOPPED, VehicleState.ILLEGAL_PARKING]:
                record.transition_to(VehicleState.MOVING, timestamp)

        return record

    def _log_violation_event(self, record: VehicleStateRecord) -> None:
        if record.violation_logged:
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bbox_str = str(record.last_bbox) if record.last_bbox is not None else "N/A"
        duration_rounded = round(record.stopped_duration, 2)

        # 1. Menulis ke log CSV lokal
        try:
            with open(self.log_file_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now_str,
                    record.vehicle_id,
                    VehicleState.ILLEGAL_PARKING,
                    duration_rounded,
                    bbox_str
                ])
        except Exception as e:
            print(f"[Error] Gagal menulis kejadian pelanggaran ke {self.log_file_path}: {e}")

        record.violation_logged = True

        # Print info ke konsol secara informatif dan estetis
        print(
            f" [VIOLATION DETECTED] | Waktu: {now_str} | ID Kendaraan: {record.vehicle_id} "
            f"| Durasi Berhenti: {duration_rounded}s | BBox: {bbox_str}"
        )

        # 2. Memicu callback eksternal jika disediakan
        if self.on_violation_callback is not None:
            try:
                self.on_violation_callback(record.vehicle_id, record.stopped_duration, record.last_bbox)
            except Exception as e:
                print(f"[Warning] Gagal mengeksekusi on_violation_callback: {e}")

    def cleanup_lost_vehicles(self, active_ids: List[int]) -> List[int]:
        active_set = set(active_ids)
        removed_ids = []

        # Mengidentifikasi ID yang tidak ada di list tracker aktif
        for vehicle_id in list(self.records.keys()):
            if vehicle_id not in active_set:
                self.records.pop(vehicle_id)
                removed_ids.append(vehicle_id)

        return removed_ids

    def get_vehicle_status(self, vehicle_id: int) -> Tuple[str, float]:
        record = self.records.get(vehicle_id)
        if record:
            return record.state, record.stopped_duration
        return VehicleState.MOVING, 0.0

    def get_all_records(self) -> Dict[int, dict]:
        return {
            vid: {
                "state": r.state,
                "stopped_duration": r.stopped_duration,
                "stop_start_time": r.stop_start_time,
                "last_bbox": r.last_bbox,
                "violation_logged": r.violation_logged
            }
            for vid, r in self.records.items()
        }

    def reset(self) -> None:
        self.records.clear()
        print("[Info] VehicleStateManager telah di-reset.")
