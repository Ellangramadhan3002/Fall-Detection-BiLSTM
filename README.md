# 🛡️ Centralized Fall Detection System (BiLSTM & RTSP CCTV)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Node.js](https://img.shields.io/badge/Node.js-16.x%2B-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-blueviolet.svg)

Sistem ini adalah solusi cerdas untuk mendeteksi kejadian jatuh (fall detection) secara *real-time* menggunakan teknologi **Deep Learning (BiLSTM)** dan **Computer Vision (MediaPipe Pose Landmarker)**. Sistem dirancang terpusat untuk terhubung langsung ke IP Camera / CCTV melalui protokol RTSP, memiliki Web Dashboard monitoring, serta otomatis mengirimkan peringatan dan *screenshot* kejadian ke WhatsApp.

---

## 🌟 Fitur Utama
1. **Real-time Fall Detection**: Mendeteksi gestur dan postur jatuh langsung dari *stream* CCTV (RTSP).
2. **Automated WhatsApp Notification**: Peringatan bahaya dikirim otomatis seketika melalui Bot WhatsApp.
3. **CCTV Auto-Scanner**: Alat bawaan untuk menemukan perangkat CCTV (IP, Port RTSP) yang tersembunyi di jaringan lokal secara otomatis.
4. **Dashboard Terpusat**: Antarmuka web modern untuk kontrol CCTV, monitoring logs, dan evaluasi riwayat jatuh.

---

## 📋 Persyaratan Sistem (Prerequisites)

Sebelum melakukan instalasi, pastikan komputer/server Anda memenuhi persyaratan berikut:
1. **Sistem Operasi**: Windows 10/11 atau Linux Ubuntu.
2. **Python**: Versi 3.9 atau lebih baru. ([Download Python](https://www.python.org/downloads/))
3. **Node.js**: Versi 16 atau lebih baru (diperlukan untuk Bot WhatsApp). ([Download Node.js](https://nodejs.org/))

---

## 🛠️ Panduan Instalasi Lengkap (Installation Manual)

Ikuti langkah-langkah di bawah ini secara berurutan. Panduan ini dirancang agar dapat diikuti oleh orang awam.

### 1. Unduh Proyek ke Komputer
Unduh repositori ini dan buka foldernya di komputer Anda.

### 2. Instalasi Dependensi Python (Sistem Utama)
Anda cukup menjalankan file *batch* yang sudah disediakan (Untuk pengguna Windows):
1. Klik ganda pada **`setup_env.bat`**. 
2. Sistem akan otomatis menginstal seluruh library AI (TensorFlow, OpenCV, MediaPipe, dll) yang dibutuhkan.

### 3. Instalasi Server WhatsApp Bot
Bot ini berfungsi untuk mengirim pesan secara otomatis ke WhatsApp jika ada yang jatuh.
Buka **Command Prompt (CMD)** atau **Terminal** di dalam folder `bot_wa_api`, lalu jalankan perintah:
```bash
npm install
```

---

## 📖 Standar Operasional Prosedur (SOP) Penggunaan

Berikut adalah panduan standar (SOP) untuk menjalankan seluruh fitur utama secara bertahap.

### TAHAP 1: Menyalakan Web Dashboard dan Detektor Jatuh
1. Buka folder proyek ini.
2. Klik ganda (Double-Click) pada file **`RUN_SERVER.bat`**.
3. Akan muncul layar hitam (Terminal) yang menunjukkan server sedang menyala. **Jangan ditutup.**
4. Buka browser web (Google Chrome/Edge/Firefox).
5. Kunjungi alamat ini: **`http://localhost:5000`**
6. Anda akan melihat halaman antarmuka (Dashboard) sistem deteksi jatuh.

### TAHAP 2: Menyalakan WhatsApp Bot (Untuk Notifikasi)
1. Buka terminal (CMD) baru. Arahkan ke folder proyek, lalu masuk ke folder `bot_wa_api`.
2. Ketik perintah: `node index.js`
3. Akan muncul **QR Code** di layar hitam (terminal).
4. Buka aplikasi WhatsApp di HP Anda -> Pilih *Linked Devices (Perangkat Taut)* -> *Scan* QR Code tersebut.
5. Bot sudah aktif! Jika sistem mendeteksi orang jatuh, bot akan otomatis mengirim pesan ke nomor target.

### TAHAP 3: Menghubungkan Kamera CCTV (RTSP)
Jika Anda tidak mengetahui alamat koneksi IP Kamera CCTV Anda, ikuti langkah ini:
1. Pastikan komputer/laptop Anda terhubung dalam satu jaringan Wi-Fi/Kabel (LAN) yang sama dengan CCTV.
2. Di dalam folder proyek, klik ganda file **`SCAN_CCTV.bat`**.
3. Program akan otomatis memindai jaringan dan memunculkan alamat IP CCTV beserta link RTSP-nya.
4. Salin link RTSP tersebut, kembali ke **Web Dashboard (`http://localhost:5000`)**.
5. Masukkan link RTSP ke kolom yang disediakan, lalu tekan **Mulai Deteksi**.

---

## 📂 Struktur Repositori

Untuk pemahaman lebih dalam, berikut adalah file-file penting yang menyusun arsitektur sistem ini:

- `05_api_server.py`: Jantung utama sistem. Menangani API, server Web, dan menghubungkan GUI dengan Engine.
- `fall_detector_engine.py`: Engine kecerdasan buatan. Mengolah frame gambar dari CCTV menjadi struktur rangka dan menjalankan deteksi BiLSTM.
- `scan_cctv.py` : Modul pemindai jaringan (Network Scanner) otomatis untuk CCTV.
- `bot_wa_api/`: Modul *micro-service* Node.js untuk integrasi API WhatsApp-Web secara gratis.
- `model/`: Folder yang menyimpan otak sistem (`fall_detection_lstm.h5` - Model BiLSTM terlatih).

---

*(Ini adalah versi Deployment/Production dari Sistem Deteksi Jatuh terpusat yang siap digunakan untuk implementasi langsung).*
