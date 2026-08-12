import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import importlib.util
import os
import time

# Konstanta dari preprocess
COCO_ORDER = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

MP_TO_COCO_IDX = {
    "Nose": 0, "Left Eye": 2, "Right Eye": 5, "Left Ear": 7, "Right Ear": 8,
    "Left Shoulder": 11, "Right Shoulder": 12, "Left Elbow": 13, "Right Elbow": 14,
    "Left Wrist": 15, "Right Wrist": 16, "Left Hip": 23, "Right Hip": 24,
    "Left Knee": 25, "Right Knee": 26, "Left Ankle": 27, "Right Ankle": 28
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "fall_detection_lstm.h5")
DATA_PATH  = os.path.join(SCRIPT_DIR, "data", "processed_data.npz")
SEQUENCE_LENGTH = 60
CONF_THRESHOLD = 0.30   # Threshold visibilitas keypoint (sesuai preprocess asli)
MIN_VISIBLE_KEYPOINTS = 6   # Minimal 6 dari 17 keypoint — cukup untuk kamera CCTV top-down

def extract_keypoints_mediapipe(landmarks, img_w, img_h) -> np.ndarray:
    row = np.zeros(17 * 3, dtype=np.float32)
    for ki, kp_name in enumerate(COCO_ORDER):
        mp_idx = MP_TO_COCO_IDX[kp_name]
        lm = landmarks[mp_idx]
        vis = lm.visibility if hasattr(lm, 'visibility') else 0.0
        if vis >= CONF_THRESHOLD:
            row[ki*3 + 0] = lm.x * img_w
            row[ki*3 + 1] = lm.y * img_h
            row[ki*3 + 2] = vis
    return row

def normalize_coords_single(mat: np.ndarray) -> np.ndarray:
    mat = mat.copy()
    n_frames = mat.shape[0]
    for fi in range(n_frames):
        xs = mat[fi, 0::3]
        ys = mat[fi, 1::3]
        valid_mask = (xs != 0) | (ys != 0)
        if valid_mask.sum() < 2:
            continue
        xv, yv = xs[valid_mask], ys[valid_mask]
        cx = (xv.min() + xv.max()) / 2.0
        cy = (yv.min() + yv.max()) / 2.0
        scale = max(xv.max() - xv.min(), yv.max() - yv.min(), 1e-5)
        
        xs_n = np.where(valid_mask, (xs - cx) / scale + 0.5, 0.0)
        ys_n = np.where(valid_mask, (ys - cy) / scale + 0.5, 0.0)
        mat[fi, 0::3] = xs_n
        mat[fi, 1::3] = ys_n
    return mat

def add_velocity_single(mat: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(mat[:, :17 * 2])
    for fi in range(1, mat.shape[0]):
        velocity[fi, 0::2] = mat[fi, 0::3] - mat[fi-1, 0::3]
        velocity[fi, 1::2] = mat[fi, 1::3] - mat[fi-1, 1::3]
    return np.concatenate([mat, velocity], axis=1)

def count_visible_keypoints(kp_row: np.ndarray) -> int:
    """Hitung berapa keypoint yang terlihat (visibility > 0) dalam satu frame."""
    count = 0
    for ki in range(17):
        vis = kp_row[ki * 3 + 2]
        if vis > CONF_THRESHOLD:
            count += 1
    return count

def is_valid_human_pose(kp_row: np.ndarray, img_w: int, img_h: int) -> bool:
    """
    Filter tambahan untuk mencegah deteksi benda bukan manusia.
    Cek dua syarat minimal:
    1. Bounding box keypoint cukup besar (>1.5% luas frame) -> bukan benda kecil/noise
    2. Ada keypoint di bagian ATAS tubuh (bahu/leher) DAN bagian BAWAH (pinggul/lutut)
       -> bukan hanya 1 sisi tubuh yang kedeteksi seperti sisi kasur
    """
    xs = [kp_row[ki*3+0] for ki in range(17) if kp_row[ki*3+2] > CONF_THRESHOLD]
    ys = [kp_row[ki*3+1] for ki in range(17) if kp_row[ki*3+2] > CONF_THRESHOLD]
    
    if len(xs) < 4:
        return False  # Terlalu sedikit keypoint
    
    # Syarat 1: Ukuran bounding box minimal 2% dari luas frame (naik dari 1.5%)
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    frame_area = img_w * img_h
    if bbox_area < frame_area * 0.02:
        return False  # Terlalu kecil, kemungkinan benda/noise
    
    # Syarat 2: MINIMAL 1 bahu terlihat (saat jatuh/rebah satu bahu bisa tertutup badan)
    any_shoulder = (kp_row[5*3+2] > CONF_THRESHOLD or kp_row[6*3+2] > CONF_THRESHOLD)
    # Syarat 3: Minimal 2 dari 4 keypoint bawah tubuh (pinggul/lutut)
    lower_count = sum(1 for ki in [11, 12, 13, 14] if kp_row[ki*3+2] > CONF_THRESHOLD)
    lower_visible = lower_count >= 2
    
    # Syarat 4: Minimal 1 keypoint wajah/kepala (hidung, mata, atau telinga)
    # Ini sangat ampuh menolak kasur/benda mati karena benda mati tidak punya wajah
    face_count = sum(1 for ki in [0, 1, 2, 3, 4] if kp_row[ki*3+2] > CONF_THRESHOLD)
    face_visible = face_count >= 1
    
    if not (any_shoulder and lower_visible and face_visible):
        # === FALLBACK UNTUK POSISI BERBARING (setelah jatuh) ===
        # Saat orang sudah rebah di kasur dari sudut kamera ATAS:
        # - Wajah sering tidak terlihat (tersembunyi di tepi frame / tertutup badan)
        # - Bahu bisa tertutup badan
        # Tapi PINGGUL hampir selalu terlihat karena merupakan titik terlebar tubuh.
        # Jika kedua pinggul terlihat + ukuran frame besar -> ini orang berbaring, bukan kasur!
        both_hips = (kp_row[11*3+2] > CONF_THRESHOLD and kp_row[12*3+2] > CONF_THRESHOLD)
        if both_hips and bbox_area >= frame_area * 0.04:  # Bbox harus cukup besar (4% frame)
            return True  # Orang berbaring, wajah tidak terlihat dari atas -> tetap valid!
        return False  # Tidak ada tubuh atas DAN bawah DAN wajah -> Benda mati (Kasur)!
            
    return True  # Lolos semua syarat -> kemungkinan besar manusia normal

def compute_sudden_drop(seq_raw, img_h) -> tuple:
    """
    Deteksi jatuh tiba-tiba dengan mengukur kecepatan turunnya pinggul (Hukum Fisika).
    Return: (is_sudden: bool, ref_torso: float, peak_velocity_ratio: float)
    peak_velocity_ratio: rasio kecepatan drop tertinggi dalam 1 frame terhadap panjang torso.
    - Jatuh sungguhan (cepat): ratio tinggi (> 0.15 per frame)
    - Tiduran dari duduk (lambat): ratio rendah (< 0.10 per frame)
    """
    left_hip_y  = seq_raw[:, 11*3 + 1]
    right_hip_y = seq_raw[:, 12*3 + 1]
    hips_y = (left_hip_y + right_hip_y) / 2.0
    
    left_shoulder_y = seq_raw[:, 5*3 + 1]
    right_shoulder_y = seq_raw[:, 6*3 + 1]
    shoulders_y = (left_shoulder_y + right_shoulder_y) / 2.0
    
    # Abaikan nilai nol
    valid = hips_y > 0
    if np.sum(valid) < 10:
        return False, img_h * 0.15, 0.0
        
    hips_y_valid = hips_y[valid]
    shoulders_y_valid = shoulders_y[valid]
    
    # 1. Hitung Panjang Torso (sebagai skala referensi jarak/depth)
    torso_lengths = hips_y_valid - shoulders_y_valid
    torso_valid = torso_lengths > 10 # Abaikan frame aneh
    if np.sum(torso_valid) > 0:
        ref_torso = np.median(torso_lengths[torso_valid])
    else:
        ref_torso = img_h * 0.15 # Fallback jika torso gagal
        
    # 2. Hitung kecepatan per-frame (np.diff) untuk menemukan LONJAKAN kecepatan tertinggi
    # Ini adalah perbedaan FUNDAMENTAL antara jatuh (cepat) vs tiduran (lambat)
    hip_diffs = np.diff(hips_y_valid)  # Perubahan posisi per frame
    positive_diffs = hip_diffs[hip_diffs > 0]  # Hanya ambil gerakan ke bawah
    
    # Kecepatan puncak tertinggi (max drop dalam 1 frame), dinormalisasi oleh panjang torso
    peak_velocity_ratio = 0.0
    if len(positive_diffs) > 0:
        peak_velocity_ratio = float(np.max(positive_diffs)) / max(ref_torso, 1.0)
    
    # [FIX-A] Perpanjang window dari 12 ke 18 frame (~0.7 detik)
    # Menangkap jatuh pelan ke kasur yang berlangsung lebih lama dari 12 frame
    window = 18
    
    # Pengecekan Teleportasi: blokir skeleton yang loncat > 150px dalam 1 frame (glitch)
    for i in range(len(hips_y) - 1):
        if hips_y[i] > 0 and hips_y[i+1] > 0:
            if abs(hips_y[i+1] - hips_y[i]) > 150:
                return False, ref_torso, 0.0
            
    max_drop = 0
    if len(hips_y_valid) > window:
        for i in range(len(hips_y_valid) - window):
            drop = hips_y_valid[i + window] - hips_y_valid[i]
            if drop > max_drop:
                max_drop = drop
    else:
        max_drop = hips_y_valid[-1] - hips_y_valid[0]
        
    # Syarat Jatuh Fisika: Anjlok total lebih dari 60% panjang punggung
    drop_threshold = ref_torso * 0.60
    
    # Catat POSISI AWAL drop (dari mana pinggul mulai anjlok)
    # Ini kunci untuk membedakan jatuh dari berdiri vs dari jongkok
    drop_start_hip_y = 0.0
    if len(hips_y_valid) > window:
        for i in range(len(hips_y_valid) - window):
            drop = hips_y_valid[i + window] - hips_y_valid[i]
            if drop > max_drop:
                drop_start_hip_y = float(hips_y_valid[i])
    elif len(hips_y_valid) > 0:
        drop_start_hip_y = float(hips_y_valid[0])
    
    is_sudden = max_drop > drop_threshold
    return is_sudden, ref_torso, peak_velocity_ratio, drop_start_hip_y


class FallDetector:
    def __init__(self, threshold=0.60):  # 60% — lebih sensitif, diimbangi gate ketat posisi
        self.threshold = threshold
        self.frame_buffer = []
        self.fall_active = False
        self.fall_just_triggered = False  # True HANYA satu frame saat foto diambil
        self.consecutive_fall_frames = 0  # Glitch filter
        self.fall_display_counter = 0     # Menahan tampilan JATUH! di UI 5 detik
        self.last_hip_y_ratio = 0.0
        self.last_peak_velocity_ratio = 0.0
        self.mediapipe_lock_counter = 0
        self.MEDIAPIPE_LOCK_FRAMES = 20  # Dinaikkan 15 → 20 frame agar lebih stabil saat jatuh cepat
        self.last_hip_y_for_bounce = 0.0
        
        # Sudden cooldown: setelah is_sudden=True, pertahankan efeknya 20 frame
        # agar base_ok tetap True saat confidence LSTM baru "tersadar" setelah drop
        self.sudden_cooldown = 0
        self.SUDDEN_COOLDOWN_FRAMES = 20
        
        # ANTI FALSE POSITIVE: Buffer voting
        self.fall_vote_buffer = []
        self.FALL_VOTE_WINDOW = 8
        self.FALL_VOTE_MIN = 5
        
        # Track orientasi tubuh: True=horizontal, False=vertikal/berdiri
        self.orientation_history = []
        self.ORIENTATION_WINDOW = 90
        self.RECENTLY_STANDING_WINDOW = 45
        
        # Track posisi pinggul historis (deteksi jongkok vs berdiri)
        self.hip_y_history = []
        self.HIP_Y_HISTORY_LEN = 120
        self.last_drop_start_hip_y = 0.0
        
        # === PEAK-CAPTURE WINDOW ===
        # Saat fall pertama kali trigger, buka jendela 30 frame.
        # Cari frame dengan confidence TERTINGGI + velocity TERTINGGI.
        # Simpan ke DB hanya 1x di titik puncak tersebut.
        self.peak_window_active = False     # Apakah jendela sedang berjalan?
        self.peak_window_counter = 0        # Hitung mundur 30 frame
        self.PEAK_WINDOW_FRAMES = 30        # ~1 detik untuk mencari puncak
        self.peak_conf = 0.0                # Confidence tertinggi yang tercatat
        self.peak_vel  = 0.0                # Velocity tertinggi yang tercatat
        self.peak_frame_ok = False          # Apakah syarat gate terpenuhi di frame puncak?
        # Gate yang harus True selama jendela agar capture valid
        self.gate_was_standing   = False    # Sebelumnya berdiri
        self.gate_was_upright    = False    # Drop dari posisi tegak (bukan jongkok)
        self.gate_reached_floor  = False    # Tubuh sampai ke lantai/kasur
        self.gate_no_bounce      = True     # Tidak ada bounce (bukan lari)
        self.gate_human_detected = False    # Manusia terdeteksi minimal 1x dalam jendela
        self.peak_confidence_locked = 0.0   # Confidence yang akan dikirim ke DB
        
        # Histori velocity 60 frame terakhir
        # Digunakan untuk menangkap kecepatan drop yang terjadi SEBELUM window dibuka
        self.vel_history = []
        
        # Load Model
        spec = importlib.util.spec_from_file_location("train_lstm", os.path.join(SCRIPT_DIR, "02_train_lstm.py"))
        train_lstm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_lstm)
        self.model = train_lstm.build_model((SEQUENCE_LENGTH, 85))
        self.model.load_weights(MODEL_PATH)
        
        # Datasets Norm
        self.train_mean = None
        self.train_std = None
        if os.path.exists(DATA_PATH):
            data = np.load(DATA_PATH)
            if "train_mean" in data:
                self.train_mean = data["train_mean"].squeeze()
                self.train_std  = data["train_std"].squeeze()
                
        # MediaPipe Configuration
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # Tetap level 1 agar tidak halu benda mati
            smooth_landmarks=True,
            min_detection_confidence=0.5, # Diturunkan agar tidak kehilangan jejak saat blur (jatuh cepat)
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_counter = 0
        self.last_prob = 0.0
        self.last_drop_vel = 0.0
        self.last_is_sudden = False
        
        @tf.function(experimental_relax_shapes=True)
        def fast_predict(x):
            return self.model(x, training=False)
        self.fast_predict = fast_predict

    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        scale_w, scale_h = 480, 360
        img_rgb_small = cv2.resize(img_rgb, (scale_w, scale_h))
        
        start_time = time.perf_counter()
        results = self.pose.process(img_rgb_small)
        
        status = "Normal"
        confidence = self.last_prob
        is_fall = False
        lstm_fall = False  # Inisialisasi awal agar tidak error
        final_fall = False  # Inisialisasi agar tidak UnboundLocalError saat buffer belum penuh
        drop_velocity = self.last_drop_vel
        body_is_horizontal = False
        was_recently_vertical = False
        valid_human_in_frame = False  # Inisialisasi: asumsikan tidak ada manusia dulu
        self.frame_counter += 1
        
        if results.pose_landmarks:
            kp_row = extract_keypoints_mediapipe(results.pose_landmarks.landmark, img_w, img_h)
            visible_kp = count_visible_keypoints(kp_row)
            
            # === CEK ORIENTASI TUBUH ===
            # Hitung bounding box dari keypoint yang terlihat
            # Saat berbaring, beberapa keypoint hilang -> turunkan threshold ke 4 jika pinggul terlihat
            hip_visible = (kp_row[11*3+2] > CONF_THRESHOLD or kp_row[12*3+2] > CONF_THRESHOLD)
            min_kp_for_orientation = 4 if hip_visible else 6
            if visible_kp >= min_kp_for_orientation:
                xs = [kp_row[ki*3+0] for ki in range(17) if kp_row[ki*3+2] > 0]
                ys = [kp_row[ki*3+1] for ki in range(17) if kp_row[ki*3+2] > 0]
                if len(xs) >= 4:
                    x_span = max(xs) - min(xs)
                    y_span = max(ys) - min(ys)
                    # Horizontal: lebar tubuh membesar relatif terhadap tinggi (karena foreshortening)
                    # Atau jarak vertikal hidung ke pinggul mengecil drastis
                    
                    aspect_ratio = x_span / max(y_span, 1.0)
                    is_wide = aspect_ratio > 0.6  # Lebih toleran untuk sudut pandang top-down
                    
                    # Cek kompresi tubuh (Hidung ke Pinggul)
                    nose_y = kp_row[0*3+1] if kp_row[0*3+2] > 0 else -1
                    left_hip_y = kp_row[11*3+1] if kp_row[11*3+2] > 0 else -1
                    right_hip_y = kp_row[12*3+1] if kp_row[12*3+2] > 0 else -1
                    
                    is_compressed = False
                    if nose_y > 0 and (left_hip_y > 0 or right_hip_y > 0):
                        hip_y = max(left_hip_y, right_hip_y)
                        # Jika jarak vertikal hidung ke pinggul sangat kecil, tubuh berarti rebah/terlipat
                        if abs(hip_y - nose_y) < (y_span * 0.4):
                            is_compressed = True
                            
                    body_is_horizontal = is_wide or is_compressed
            
            # === FILTER: Cek apakah ini benar-benar manusia ===
            valid_human_in_frame = is_valid_human_pose(kp_row, img_w, img_h)
            
            # HANYA simpan history orientasi jika pose VALID (bukan kasur/benda)
            # Ini mencegah kasur mengkontaminasi data orientation_history
            if valid_human_in_frame:
                self.orientation_history.append(body_is_horizontal)
            else:
                # Benda terdeteksi, catat sebagai "tidak ada orang" di history
                self.orientation_history.append(False)
            
            if len(self.orientation_history) > self.ORIENTATION_WINDOW:
                self.orientation_history = self.orientation_history[-self.ORIENTATION_WINDOW:]
            
            # Cek apakah baru-baru ini berdiri (vertikal) sebelum jatuh
            if len(self.orientation_history) >= 15:
                vertical_count = sum(1 for h in self.orientation_history if not h)
                was_recently_vertical = vertical_count >= 15
            
            # Gambar skeleton (tetap digambar agar user bisa melihat apa yang terdeteksi)
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 229, 255), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 255, 88), thickness=2, circle_radius=2)
            )
            
            # Masukkan keypoint ke buffer
            if results.pose_landmarks:
                self.frame_buffer.append(kp_row)
                # [FIX-D] Saat skeleton valid terdeteksi, aktifkan/perbarui lock counter
                if valid_human_in_frame:
                    self.mediapipe_lock_counter = self.MEDIAPIPE_LOCK_FRAMES
        else:
            # [FIX-D] MediaPipe kehilangan skeleton (blur saat jatuh cepat)
            # Pertahankan valid_human selama MEDIAPIPE_LOCK_FRAMES frame
            # agar confidence tidak anjlok ke 1% tepat saat orang mendarat
            if self.mediapipe_lock_counter > 0:
                self.mediapipe_lock_counter -= 1
                valid_human_in_frame = True  # Paksa valid selama cooldown
            # Duplikat pose terakhir (jangan nol) agar LSTM tidak rusak
            if len(self.frame_buffer) > 0:
                self.frame_buffer.append(self.frame_buffer[-1])
            else:
                self.frame_buffer.append(np.zeros(17 * 3, dtype=np.float32))
            
        if len(self.frame_buffer) > SEQUENCE_LENGTH:
            self.frame_buffer = self.frame_buffer[-SEQUENCE_LENGTH:]
            
        if len(self.frame_buffer) == SEQUENCE_LENGTH:
            seq = np.array(self.frame_buffer, dtype=np.float32)
            if self.frame_counter % 4 == 0:
                is_sudden, ref_torso, peak_vel_ratio, drop_start_hip = compute_sudden_drop(seq, img_h)
                self.last_is_sudden = is_sudden
                self.last_ref_torso = ref_torso
                self.last_peak_velocity_ratio = peak_vel_ratio
                self.last_drop_start_hip_y = drop_start_hip
                
                seq_reshaped = seq.reshape(SEQUENCE_LENGTH, -1)
                seq_feat = seq_reshaped.copy()
                seq_norm = normalize_coords_single(seq_feat)
                seq_full = add_velocity_single(seq_norm)
                
                if self.train_mean is not None:
                    seq_full = (seq_full - self.train_mean) / (self.train_std + 1e-8)
                    
                X = seq_full[np.newaxis]
                
                pred_tensor = self.fast_predict(X)
                prob = float(pred_tensor.numpy().flatten()[0])
                self.last_prob = prob
                
                left_hip_y  = seq[:, 11*3 + 1]
                right_hip_y = seq[:, 12*3 + 1]
                hips_y = (left_hip_y + right_hip_y) / 2.0
                valid = hips_y > 0
                if np.sum(valid) > 1:
                    self.last_drop_vel = float(hips_y[valid][-1] - hips_y[valid][0])

            confidence = self.last_prob
            drop_velocity = self.last_drop_vel
            
            # Update riwayat posisi pinggul (dipakai untuk deteksi jongkok vs berdiri)
            kp_last = seq[-1]
            lh_y = kp_last[11*3+1]; rh_y = kp_last[12*3+1]
            if lh_y > 0 and rh_y > 0:
                self.hip_y_history.append((lh_y + rh_y) / 2.0)
            elif lh_y > 0:
                self.hip_y_history.append(lh_y)
            elif rh_y > 0:
                self.hip_y_history.append(rh_y)
            if len(self.hip_y_history) > self.HIP_Y_HISTORY_LEN:
                self.hip_y_history = self.hip_y_history[-self.HIP_Y_HISTORY_LEN:]

            # Update riwayat velocity (60 frame) untuk seed peak_vel saat window dibuka
            peak_vel_now = getattr(self, 'last_peak_velocity_ratio', 0.0)
            self.vel_history.append(peak_vel_now)
            if len(self.vel_history) > 60:
                self.vel_history = self.vel_history[-60:]

            # === SYARAT JATUH ===
            # WAJIB dipenuhi SEMUA:
            # 1. BiLSTM confidence >= threshold
            # 2. Pose valid (bukan benda)
            # 3. Kecepatan anjlok tiba-tiba (is_sudden_drop)
            # 4. PINGGUL SUDAH DI LEVEL LANTAI (> 45% dari tinggi frame ke bawah)
            #    -> Ini kunci utama: membungkuk TIDAK membuat pinggul ke lantai!
            is_sudden = getattr(self, 'last_is_sudden', False)
            peak_vel = getattr(self, 'last_peak_velocity_ratio', 0.0)
            
            # === CEK APAKAH TUBUH BENAR-BENAR DI LANTAI/KASUR ===
            kp_now = seq[-1]
            left_hip_y_now  = kp_now[11*3 + 1]
            right_hip_y_now = kp_now[12*3 + 1]
            hip_ys_now = [y for y in [left_hip_y_now, right_hip_y_now] if y > 0]
            
            is_on_floor = False
            if hip_ys_now:
                avg_hip_y_now = sum(hip_ys_now) / len(hip_ys_now)
                ankles_y = [kp_now[15*3+1], kp_now[16*3+1]]
                ankles_c = [kp_now[15*3+2], kp_now[16*3+2]]
                knees_y  = [kp_now[13*3+1], kp_now[14*3+1]]
                knees_c  = [kp_now[13*3+2], kp_now[14*3+2]]
                valid_ankles = [y for y, c in zip(ankles_y, ankles_c) if c > 0.3]
                valid_knees = [y for y, c in zip(knees_y, knees_c) if c > 0.3]
                lowest_leg_y = avg_hip_y_now
                if valid_ankles:
                    lowest_leg_y = max(valid_ankles)
                elif valid_knees:
                    lowest_leg_y = max(valid_knees)
                leg_vertical_span = lowest_leg_y - avg_hip_y_now
                ref_t = getattr(self, 'last_ref_torso', img_h * 0.15)
                is_on_floor = leg_vertical_span < (ref_t * 0.8)
            
            # === CEK APAKAH BARU SAJA BERDIRI (dalam 45 frame / ~1.5 detik terakhir) ===
            recently_standing_window = self.orientation_history[-self.RECENTLY_STANDING_WINDOW:]
            vertical_in_recent = sum(1 for h in recently_standing_window if not h)
            # [FIX-C] Turunkan dari 10 ke 6: lebih mudah dinyatakan "baru berdiri"
            # agar jatuh pelan dari posisi berdiri tidak kehilangan flag ini
            was_recently_standing = vertical_in_recent >= 6
            
            # [FIX-B] BOUNCE-BACK FILTER (Anti berlari dianggap jatuh)
            # Saat berlari, setelah pinggul "drop" cepat, pinggul langsung NAIK lagi (bounce)
            # Saat jatuh sungguhan, setelah drop, pinggul TETAP di bawah (tidak naik)
            # Cek: dalam 8 frame terakhir, apakah pinggul naik kembali > 30% torso setelah drop?
            hip_bounce_detected = False
            if peak_vel > 0.12:  # Hanya cek bounce untuk fast drop
                hips_recent = []
                for fi in range(max(0, len(self.frame_buffer) - 8), len(self.frame_buffer)):
                    lh = self.frame_buffer[fi][11*3+1]
                    rh = self.frame_buffer[fi][12*3+1]
                    if lh > 0 or rh > 0:
                        hips_recent.append((lh + rh) / max(lh > 0 and rh > 0, 1) if (lh > 0 and rh > 0) else max(lh, rh))
                if len(hips_recent) >= 4:
                    # Cari drop (max) lalu cek apakah setelah drop ada kenaikan signifikan
                    max_hip_y = max(hips_recent)
                    last_hip_y = hips_recent[-1]
                    ref_t = getattr(self, 'last_ref_torso', img_h * 0.15)
                    # Jika pinggul sudah naik kembali > 20% dari torso setelah mencapai titik terendah
                    if (max_hip_y - last_hip_y) > (ref_t * 0.20):
                        hip_bounce_detected = True
            
            # ================================================================
            # === DECISION TREE: JATUH vs TIDURAN vs BERLARI ===
            # ================================================================
            base_ok = (confidence >= self.threshold) and valid_human_in_frame and is_sudden
            posture_ok = is_on_floor or body_is_horizontal
            
            # === HIP BASELINE CHECK: Bedakan jongkok vs berdiri ===
            drop_start = getattr(self, 'last_drop_start_hip_y', 0.0)
            was_upright_before_fall = True
            if len(self.hip_y_history) >= 30 and drop_start > 0:
                min_hip_y_recent = min(self.hip_y_history[-90:]) if len(self.hip_y_history) >= 90 else min(self.hip_y_history)
                margin = img_h * 0.18
                was_upright_before_fall = drop_start <= (min_hip_y_recent + margin)
            
            lstm_fall = False
            if base_ok and posture_ok:
                if hip_bounce_detected:
                    lstm_fall = False  # Berlari/melompat
                elif not was_upright_before_fall:
                    lstm_fall = False  # Jongkok -> rebahan, bukan jatuh
                elif peak_vel > 0.15:
                    lstm_fall = True   # Jatuh cepat dari berdiri
                elif peak_vel > 0.08:
                    lstm_fall = was_recently_standing  # Jatuh lambat harus dari posisi berdiri
                else:
                    lstm_fall = False  # Kecepatan drop sangat rendah (< 0.08) -> gerakan tiduran terkontrol
            
            # === SUDDEN COOLDOWN: Pertahankan efek is_sudden selama N frame ===
            # Masalah: saat jatuh cepat, is_sudden=True saat drop, tapi confidence LSTM
            # baru mencapai puncak 20 frame kemudian saat orang sudah di lantai.
            # Cooldown memastikan base_ok tetap True selama "ekor" setelah drop.
            if is_sudden:
                self.sudden_cooldown = self.SUDDEN_COOLDOWN_FRAMES
            elif self.sudden_cooldown > 0:
                self.sudden_cooldown -= 1
            effective_sudden = is_sudden or (self.sudden_cooldown > 0)
            
            # --- GLITCH FILTER (Anti-UDP Packet Loss) ---
            base_ok_for_fall = (confidence >= self.threshold) and valid_human_in_frame and effective_sudden
            posture_ok = is_on_floor or body_is_horizontal
            
            # Recalculate lstm_fall with effective_sudden (for display/consecutive frames)
            if base_ok_for_fall and posture_ok:
                if hip_bounce_detected:
                    pass  # Keep lstm_fall as is (already set above)
                # lstm_fall already computed above with original base_ok; just re-check
            
            # Override: jika effective_sudden mengubah keputusan
            if not lstm_fall and base_ok_for_fall and posture_ok:
                if not hip_bounce_detected and was_upright_before_fall:
                    if peak_vel > 0.15:
                        lstm_fall = True
                    elif peak_vel > 0.08:
                        lstm_fall = was_recently_standing
            
            if lstm_fall:
                self.consecutive_fall_frames += 1
            else:
                self.consecutive_fall_frames = 0
                
            final_fall = self.consecutive_fall_frames >= 2
            
            # === DEBUG PRINT KE TERMINAL ===
            # Hanya print jika ada gerakan anjlok (is_sudden) atau AI agak yakin (conf > 0.3)
            # Ini akan membantu kita melihat kenapa jatuh depan/belakang gagal
            if is_sudden or confidence > 0.25:
                print(f"[DEBUG] Conf:{confidence*100:.1f}% | Sudden:{is_sudden} | PeakVel:{peak_vel:.3f} | Upright:{was_upright_before_fall} | Bounce:{hip_bounce_detected} | Floor:{is_on_floor} | Horiz:{body_is_horizontal} | RS:{was_recently_standing}")
            
            # ================================================================
            # === PEAK-CAPTURE WINDOW (Ganti logika delay capture lama) ===
            # ================================================================
            # Saat final_fall pertama True: BUKA jendela 30 frame.
            # Dalam jendela: lacak puncak confidence & velocity, kumpulkan gate.
            # Di akhir jendela: jika semua gate OK → trigger capture 1x.
            
            if final_fall and not self.peak_window_active and self.fall_display_counter == 0:
                # Buka jendela baru
                self.peak_window_active    = True
                self.peak_window_counter   = self.PEAK_WINDOW_FRAMES
                # SEED dari histori: jangan mulai dari 0!
                # Drop bisa terjadi SEBELUM window dibuka, tangkap dengan histori
                self.peak_conf  = confidence  # Mulai dari confidence saat ini
                self.peak_vel   = max(self.vel_history) if self.vel_history else peak_vel
                self.peak_confidence_locked = confidence
                # Reset gate
                self.gate_was_standing  = False
                self.gate_was_upright   = False
                self.gate_reached_floor = False
                self.gate_no_bounce     = True
                self.gate_human_detected = False
                print(f"[WINDOW BUKA] seed_conf:{confidence*100:.1f}% | seed_vel:{self.peak_vel:.3f} | vel_hist_max:{max(self.vel_history) if self.vel_history else 0:.3f}")
            
            self.fall_just_triggered = False  # Default: tidak capture
            
            if self.peak_window_active:
                self.peak_window_counter -= 1
                
                # Update gate secara kumulatif (sekali True, tetap True)
                if was_recently_standing:
                    self.gate_was_standing = True
                if was_upright_before_fall:
                    self.gate_was_upright = True
                if posture_ok:
                    self.gate_reached_floor = True
                if hip_bounce_detected:
                    self.gate_no_bounce = False
                if valid_human_in_frame:
                    self.gate_human_detected = True
                
                # === FIX BUG 1: Track confidence & velocity SECARA TERPISAH ===
                # Confidence diupdate kapanpun ada yang lebih tinggi (tidak perlu tunggu vel>0)
                # Velocity ditrack terpisah, hanya saat positif (pinggul turun)
                if confidence > self.peak_conf:
                    self.peak_conf = confidence
                    self.peak_confidence_locked = confidence
                # Velocity: hanya update jika positif DAN lebih tinggi dari sebelumnya
                if peak_vel > 0 and peak_vel > self.peak_vel:
                    self.peak_vel = peak_vel
                
                # Jendela selesai?
                if self.peak_window_counter <= 0:
                    self.peak_window_active = False  # Tutup jendela
                    
                    # === GATE AKHIR: Semua syarat harus terpenuhi ===
                    all_gates_ok = (
                        self.peak_conf >= self.threshold and  # BiLSTM >= 60%
                        self.peak_vel  > 0.08 and            # [FIX] Kecepatan anjlok minimal 0.08 untuk menolak rebahan santai
                        self.gate_was_standing and           # Sebelumnya berdiri
                        self.gate_was_upright and            # Drop dari posisi tegak, bukan jongkok
                        self.gate_reached_floor and          # Tubuh sampai ke lantai/kasur
                        self.gate_no_bounce and              # Bukan gerakan lari/melompat
                        self.gate_human_detected             # Manusia terdeteksi dalam jendela
                    )
                    
                    if all_gates_ok:
                        self.fall_just_triggered = True  # CAPTURE! Simpan ke DB
                        print(f"[CAPTURE] GATE OK → Conf:{self.peak_conf*100:.1f}% | Vel:{self.peak_vel:.3f} | Stand:{self.gate_was_standing} | Upright:{self.gate_was_upright} | Floor:{self.gate_reached_floor}")
                    else:
                        print(f"[CAPTURE DITOLAK] Conf:{self.peak_conf*100:.1f}% | Vel:{self.peak_vel:.3f} | Stand:{self.gate_was_standing} | Upright:{self.gate_was_upright} | Floor:{self.gate_reached_floor} | NoBounce:{self.gate_no_bounce}")
                    
                    # Kunci confidence untuk ditampilkan di UI
                    confidence = self.peak_confidence_locked if self.peak_confidence_locked > 0 else confidence
            
        # --- LOGIKA UI: Alert JATUH berkedip 5 detik, video TIDAK pernah berhenti ---
        if final_fall:
            # Jatuh baru terdeteksi: set/reset timer display 5 detik (100 frame @ ~20fps)
            self.fall_display_counter = 100
        
        if self.fall_display_counter > 0:
            status = "FALL"
            is_fall = True
            self.fall_display_counter -= 1
        else:
            status = "Normal"
            is_fall = False
        
        end_time = time.perf_counter()
        fps = 1.0 / (end_time - start_time + 1e-5)
        
        return {
            "status": status,
            "confidence": float(confidence),
            "is_fall": bool(is_fall),
            "fall_just_triggered": bool(self.fall_just_triggered),
            "body_is_horizontal": bool(body_is_horizontal),
            "drop_velocity": float(drop_velocity),
            "fps": float(fps),
            "frame": int(self.frame_counter)
        }
