import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|timeout;5000000"


import cv2
import time
import queue
import threading
import sys
from datetime import datetime

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import engine, SessionLocal, FallEvent, SetupSetting
from fall_detector_engine import FallDetector
from pydantic import BaseModel
from typing import List

# ============================================================
# MEDIA STORAGE: Disimpan di luar folder project agar ringan
# Folder: E:\FALL_DETECTION_MEDIA\YYYY-MM\
# ============================================================
MEDIA_BASE = r"E:\FALL_DETECTION_MEDIA"

def get_media_dir():
    """Buat dan kembalikan folder media bulan ini (auto-organisasi per bulan)."""
    from datetime import datetime
    month_folder = datetime.now().strftime("%Y-%m")
    path = os.path.join(MEDIA_BASE, month_folder)
    os.makedirs(path, exist_ok=True)
    return path

os.makedirs(MEDIA_BASE, exist_ok=True)

app = FastAPI(title="Fall Detection System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_URL = "D:/SMSTR 6/Lansia/orang tua/archive/dataset_action_split/test/Lying Down/video_1.avi"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

RECORDINGS_DIR = "E:\\FALL_DETECTION_RECORDINGS"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

class CameraStream:
    def __init__(self, camera_id: str, url: str):
        self.camera_id = camera_id
        self.url = url
        self.detector = FallDetector(threshold=0.70)
        self.cap = None  # JANGAN diinisialisasi di sini agar tidak memblokir main thread API!
        self.current_frame = None
        self.current_metadata = {}
        self.running = True
        self.detection_active = True  # Bisa di-pause via tombol Hentikan Semua
        self.enable_db_save = True    # Bisa dimatikan saat evaluasi/testing
        self.fail_count = 0 # Robust EOF/Disconnect tracker
        self.needs_reconnect = True # SET TRUE di awal agar _update yg membuka koneksi kamera
        
        self.frame_delay = 0.0
        self.source_fps = 20.0
        
        self.last_fall_time = 0  # kept for compatibility, not used for DB save
        self.last_wa_sent_time = 0.0  # COOLDOWN WHATSAPP
        
        # Fitur Rekam Video Bersih
        self.is_recording = False
        self.video_writer = None
        self.recording_path = ""
        self.recording_lock = threading.Lock()
        self.last_record_time = 0.0  # Kontrol kecepatan tulis frame rekaman
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Producer-Consumer: AI diproses di thread terpisah agar video TIDAK PERNAH FREEZE
        self.ai_queue = queue.Queue(maxsize=1) # maxsize=1: AI selalu proses frame TERBARU
        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()

    def _is_valid_frame(self, frame) -> bool:
        """Tolak frame corrupt: cek abu-abu polos (biasa terjadi jika RTSP drop packet)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 10.0:
            return False
        return True

    def start_recording(self):
        """Mulai merekam video bersih (tanpa skeleton) ke disk E."""
        with self.recording_lock:
            if self.is_recording:
                return {"status": "already_recording", "path": self.recording_path}
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.camera_id.replace(' ', '_')}_{timestamp}.avi"
            self.recording_path = os.path.join(RECORDINGS_DIR, filename)
            
            # Gunakan FPS asli kamera agar hasil rekaman SAMA PERSIS dengan live stream
            # Bukan hardcode 20.0 yang menyebabkan rekaman lebih cepat/lambat dari aslinya
            record_fps = self.source_fps
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(self.recording_path, fourcc, record_fps, (640, 480))
            self.is_recording = True
            print(f"[REC] Mulai merekam ke: {self.recording_path} | FPS: {record_fps:.1f}")
            return {"status": "started", "path": self.recording_path}

    def stop_recording(self):
        """Hentikan perekaman dan simpan file."""
        with self.recording_lock:
            if not self.is_recording:
                return {"status": "not_recording"}
            
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            path = self.recording_path
            self.recording_path = ""
            print(f"[REC] Rekaman disimpan: {path}")
            return {"status": "stopped", "path": path}

    def _update(self):
        """PRODUCER: Hanya membaca frame. TIDAK PERNAH diblokir oleh AI."""
        while self.running:
            if self.needs_reconnect:
                if self.cap:
                    self.cap.release()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.needs_reconnect = False
                self.fail_count = 0
                
                # Setup FPS (dipindahkan dari __init__)
                is_rtsp = str(self.url).lower().startswith('rtsp://') or str(self.url).lower().startswith('http')
                raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if not is_rtsp:
                    if raw_fps <= 15 or raw_fps > 35:
                        raw_fps = 25.0
                    self.frame_delay = 1.0 / raw_fps
                else:
                    self.frame_delay = 0.0
                
                if raw_fps <= 0 or raw_fps > 60:
                    self.source_fps = 20.0
                else:
                    self.source_fps = raw_fps
                    
                time.sleep(0.5)
                continue
                
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            success, frame = self.cap.read()
            if not success:
                self.fail_count += 1
                if self.fail_count > 5:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    self.fail_count = 0
                time.sleep(0.05)
                continue
            else:
                self.fail_count = 0
            
            frame = cv2.resize(frame, (640, 480))
            
            if not self._is_valid_frame(frame):
                continue
            
            # Simpan frame bersih untuk rekaman
            # Kontrol kecepatan tulis agar rekaman berjalan SAMA PERSIS dengan live stream
            # Tanpa ini, frame RTSP yang datang terlalu cepat akan membuat video rekaman fast-forward
            with self.recording_lock:
                if self.is_recording and self.video_writer:
                    now = time.perf_counter()
                    record_interval = 1.0 / self.source_fps
                    if (now - self.last_record_time) >= record_interval:
                        self.video_writer.write(frame)
                        self.last_record_time = now
            
            # Kirim ke antrian AI (non-blocking). Jika AI sedang sibuk, frame dibuang.
            # JANGAN tulis current_frame di sini - hanya _ai_worker yang menulis
            # agar skeleton tidak kedip-kedip (flicker)!
            try:
                self.ai_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
            
            # Kontrol kecepatan baca frame untuk input file video
            # Agar video tidak diputar terlalu cepat (melebihi FPS aslinya)
            if self.frame_delay > 0:
                time.sleep(self.frame_delay) # AI sedang sibuk, lewati frame ini, stream tetap jalan

    def _ai_worker(self):
        """CONSUMER: Jalankan AI di thread terpisah. Video tidak pernah menunggu ini."""
        while self.running:
            try:
                try:
                    frame = self.ai_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if self.detection_active:
                    res = self.detector.process_frame(frame) # Gambar skeleton di atas frame
                else:
                    res = {"status": "Paused", "confidence": 0.0, "is_fall": False,
                           "fall_just_triggered": False, "body_is_horizontal": False,
                           "drop_velocity": 0.0, "fps": 0.0, "frame": 0}
                
                current_f = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap else 0
                total_f = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0
                
                # [FIX] Jika stream berupa RTSP (Live), OpenCV sering mengembalikan nilai minus acak.
                if total_f <= 0 or total_f > 1000000000:
                    res["video_frame"] = f"{current_f}/Live"
                else:
                    res["video_frame"] = f"{current_f}/{total_f}"
                res["is_recording"] = self.is_recording
                self.current_metadata = res
                
                # Update current_frame dengan frame yang sudah diproses AI
                # (Ini bisa 10-15 FPS, tidak apa-apa karena _update jalan terus)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    self.current_frame = buffer.tobytes()
                
                # Simpan ke DB hanya saat fall_just_triggered (1 foto, di thread terpisah)
                if self.detection_active and self.enable_db_save and res.get("fall_just_triggered", False):
                    threading.Thread(
                        target=self._save_to_db,
                        args=(frame.copy(), res),
                        daemon=True
                    ).start()
            except Exception as e:
                print(f"[ERROR in _ai_worker] {e}")
                import traceback
                traceback.print_exc()



    def _save_to_db(self, frame, res):
        db = SessionLocal()
        try:
            setting = db.query(SetupSetting).first()
            save_mode = setting.media_save_mode if setting else "image"
            
            # Gunakan folder media bulan ini di E:\FALL_DETECTION_MEDIA\YYYY-MM\
            media_dir = get_media_dir()
            filename = f"{self.camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            media_path = ""
            
            # Simpan sebagai foto (gambar momen jatuh)
            filepath = os.path.join(media_dir, f"{filename}.jpg")
            cv2.imwrite(filepath, frame)
            media_path = filepath
                
            new_event = FallEvent(
                camera_id=self.camera_id,
                status="FALL",
                confidence=res["confidence"],
                frames_analyzed=60,
                media_type=save_mode,
                media_path=media_path
            )
            db.add(new_event)
            db.commit()
            print(f"\n[ALERT] JATUH TERDETEKSI KAMERA {self.camera_id}")
            print(f"[SAVED] Foto disimpan: {filepath}")
            
            # --- MULAI: Integrasi Bot WA Armbian ---
            current_time = time.time()
            if current_time - self.last_wa_sent_time > 60: # Cooldown 60 detik
                self.last_wa_sent_time = current_time
                threading.Thread(
                    target=self.send_wa_notification_api, 
                    args=(filepath, res["confidence"]), 
                    daemon=True
                ).start()
            else:
                print("[WA] Menunda WA (Cooldown 60 detik untuk mencegah spam).")
            # --- AKHIR: Integrasi Bot WA Armbian ---
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Gagal menyimpan ke database: {e}")
        finally:
            db.close()

    def send_wa_notification_api(self, image_path, confidence):
        """Fungsi untuk mengirim gambar + teks ke Bot Node.js di server Armbian"""
        import requests
        try:
            # Menggunakan IP Asli Armbian untuk menghindari blokir router
            bot_api_url = "https://ellang.rrlabs.web.id/send" 

            # TODO: Ganti dengan nomor WhatsApp tujuan Anda
            phone_number = "08980509128"
            
            caption = (
                f"⚠️ *PERINGATAN: KEJADIAN JATUH* ⚠️\n"
                f"Kamera: {self.camera_id}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Status: FALL\n"
                f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Tolong segera periksa lokasi!"
            )
            
            # Buka file gambar dan kirim via HTTP POST Form-Data
            with open(image_path, "rb") as img_file:
                files = {"image": img_file}
                data = {
                    "phone": phone_number,
                    "caption": caption
                }
                print(f"[WA] Mengirim gambar ke Bot Server {bot_api_url}...")
                response = requests.post(bot_api_url, data=data, files=files, timeout=15)
                
            if response.status_code == 200:
                print("[WA] ✅ Notifikasi berhasil dikirim via Bot Server!")
            else:
                print(f"[WA] ❌ Bot Server menolak: {response.text}")
                
        except Exception as e:
            print(f"[WA] ❌ Error menghubungi Bot Server (Pastikan Node.js berjalan): {e}")

    def get_frame(self):
        return self.current_frame

    def update_url(self, new_url: str):
        print(f"[INFO] Merubah sumber kamera {self.camera_id} ke: {new_url}")
        self.url = new_url
        self.needs_reconnect = True

cam_managers = {
    "1": CameraStream("Kamera 1", VIDEO_URL)
}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

def generate_mjpeg(cam_id: str):
    while True:
        if cam_id not in cam_managers:
            time.sleep(1)
            continue
            
        frame = cam_managers[cam_id].get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # Batasi ke ~30 FPS agar thread _update tidak kelaparan CPU

@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str):
    return StreamingResponse(generate_mjpeg(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def get_camera_status():
    status_dict = {}
    for cam_id, manager in cam_managers.items():
        status_dict[cam_id] = manager.current_metadata
    return status_dict

@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    """Ambil SEMUA riwayat jatuh dari database, diurutkan dari terbaru."""
    logs = db.query(FallEvent).order_by(FallEvent.timestamp.desc()).all()
    return logs

@app.get("/open-media")
def open_media_file(path: str):
    """Buka file media (foto/video) langsung di Windows menggunakan aplikasi default."""
    import urllib.parse
    decoded_path = urllib.parse.unquote(path)
    if not os.path.exists(decoded_path):
        return {"error": f"File tidak ditemukan: {decoded_path}"}
    try:
        os.startfile(decoded_path)
        return {"ok": True, "path": decoded_path}
    except Exception as e:
        return {"error": str(e)}

@app.post("/detection/start")
def detection_start():
    """Aktifkan deteksi AI (Mulai Semua)."""
    for manager in cam_managers.values():
        manager.detection_active = True
    return {"message": "Deteksi AI diaktifkan", "active": True}

@app.post("/detection/stop")
def detection_stop():
    """Hentikan sementara deteksi AI (Hentikan Semua)."""
    for manager in cam_managers.values():
        manager.detection_active = False
    return {"message": "Deteksi AI dihentikan sementara", "active": False}

@app.post("/recording/start")
def recording_start():
    """Mulai merekam video bersih ke disk E."""
    results = {}
    for cam_id, manager in cam_managers.items():
        results[cam_id] = manager.start_recording()
    return {"status": "started", "details": results}

@app.post("/recording/stop")
def recording_stop():
    """Hentikan perekaman dan simpan file."""
    results = {}
    path = ""
    for cam_id, manager in cam_managers.items():
        res = manager.stop_recording()
        results[cam_id] = res
        if res.get("status") == "stopped" and not path:
            path = res.get("path", "")
    return {"status": "stopped", "path": path, "details": results}

@app.post("/recording/toggle_db")
def toggle_db_save():
    """Matikan/nyalakan proses simpan jatuh ke database."""
    new_status = True
    if "1" in cam_managers:
        new_status = not cam_managers["1"].enable_db_save
        
    for manager in cam_managers.values():
        manager.enable_db_save = new_status
        
    status_msg = "AKTIF" if new_status else "MATI"
    return {"message": f"Penyimpanan Database {status_msg}", "db_save_active": new_status}

@app.post("/settings/media_mode")
def update_media_mode(mode: str, db: Session = Depends(get_db)):
    if mode not in ["image", "video"]:
        return {"error": "Pilih 'image' atau 'video'"}
    
    setting = db.query(SetupSetting).first()
    if not setting:
        setting = SetupSetting(media_save_mode=mode)
        db.add(setting)
    else:
        setting.media_save_mode = mode
    db.commit()
    return {"message": "Setting tersimpan", "mode": mode}

class CameraSettingsPayload(BaseModel):
    urls: List[str]

@app.post("/settings/camera")
def update_camera_url(payload: CameraSettingsPayload):
    for idx, url in enumerate(payload.urls):
        cam_id = str(idx + 1)
        
        if not url.strip():
            # Jika input kosong, hentikan kamera tersebut agar tidak menjadi 'ghost stream'
            if cam_id in cam_managers:
                print(f"[INFO] Menghentikan Kamera {cam_id} karena input URL dikosongkan.")
                manager = cam_managers.pop(cam_id)
                manager.running = False
                manager.cap.release()
            continue
            
        if cam_id in cam_managers:
            if cam_managers[cam_id].url != url:
                cam_managers[cam_id].update_url(url)
        else:
            cam_managers[cam_id] = CameraStream(f"Kamera {cam_id}", url)
            if "1" in cam_managers:
                cam_managers[cam_id].detection_active = cam_managers["1"].detection_active
                cam_managers[cam_id].enable_db_save = cam_managers["1"].enable_db_save
                
    return {"message": "Kamera berhasil diperbarui", "urls": payload.urls}

@app.delete("/logs")
def clear_all_logs(db: Session = Depends(get_db)):
    import shutil
    
    logs = db.query(FallEvent).all()
    
    for log in logs:
        if log.media_path and os.path.exists(log.media_path):
            try:
                os.remove(log.media_path)
            except Exception as e:
                print(f"[WARNING] Gagal menghapus file {log.media_path}: {e}")
                
    db.query(FallEvent).delete()
    db.commit()
    
    return {"message": "✅ Seluruh riwayat Database & Media berhasil dihapus!"}

if __name__ == "__main__":
    import uvicorn
    print("=========================================================")
    print("\n   ✅ API Server & DASHBOARD Siap!")
    print("   🌐 BUKA ALAMAT INI DI BROWSER: http://localhost:8000")
    print("\n=========================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
