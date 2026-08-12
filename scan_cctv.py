"""
=================================================================
  CCTV IP SCANNER - Sistem Deteksi Jatuh BiLSTM
  Oleh: Ellang Ramadhan
  Fungsi: Memindai jaringan secara otomatis untuk menemukan
          alamat IP kamera CCTV yang mendukung protokol RTSP
=================================================================
Cara pakai:
  1. Pastikan Laptop terhubung ke WiFi yang sama dengan CCTV
  2. Jalankan: python scan_cctv.py
  3. Program akan otomatis mendeteksi subnet jaringan Anda
  4. Tunggu hasil scan (5-15 detik)
=================================================================
"""

import socket
import concurrent.futures
import time
import subprocess
import platform
import sys

# ─────────────────────────────────────────────
# KONFIGURASI - Ubah jika diperlukan
# ─────────────────────────────────────────────
RTSP_PORT     = 554       # Port standar RTSP untuk kamera IP
ONVIF_PORT    = 80        # Port ONVIF (beberapa CCTV)
HTTP_PORT     = 8080      # Port web kamera
SCAN_TIMEOUT  = 0.8       # Timeout per IP (detik) - perbesar jika jaringan lambat
MAX_WORKERS   = 100       # Jumlah thread paralel

# Format RTSP yang umum digunakan berbagai merek kamera
RTSP_FORMATS = [
    "rtsp://{user}:{pwd}@{ip}:554/live/ch00_1",   # Upupin / Reolink
    "rtsp://{user}:{pwd}@{ip}:554/stream1",        # Hikvision / Dahua
    "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0",  # Dahua
    "rtsp://{user}:{pwd}@{ip}:554/h264Preview_01_main",  # Reolink
    "rtsp://{user}:{pwd}@{ip}:554/video1",         # Axis
    "rtsp://{user}:{pwd}@{ip}:554/1",              # Generic
    "rtsp://{user}:{pwd}@{ip}:554/",               # Generic fallback
]

# Kombinasi username/password umum kamera IP
COMMON_CREDENTIALS = [
    ("admin",  "admin"),
    ("admin",  ""),
    ("admin",  "12345"),
    ("admin",  "123456"),
    ("admin",  "password"),
    ("root",   "root"),
    ("root",   "admin"),
    ("user",   "user"),
    ("admin",  "en3a3p"),    # Upupin khusus
]


def get_local_subnet():
    """Deteksi otomatis subnet jaringan laptop saat ini."""
    try:
        # Koneksi dummy untuk mendapatkan IP lokal
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.split(".")
        subnet = f"{parts[0]}.{parts[1]}.{parts[2]}."
        return subnet, local_ip
    except Exception:
        return "192.168.1.", "Tidak diketahui"


def check_port(ip, port, timeout=None):
    """Cek apakah port tertentu terbuka di IP yang diberikan."""
    if timeout is None:
        timeout = SCAN_TIMEOUT
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((ip, port)) == 0
    except Exception:
        return False


def try_rtsp_connect(ip, credentials=None):
    """
    Coba koneksi RTSP dengan berbagai format URL dan kredensial.
    Mengembalikan URL yang berhasil atau None.
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    if credentials is None:
        credentials = COMMON_CREDENTIALS

    for user, pwd in credentials:
        for fmt in RTSP_FORMATS:
            url = fmt.format(user=user, pwd=pwd, ip=ip)
            if has_cv2:
                try:
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            return url, user, pwd
                except Exception:
                    pass
            else:
                # Tanpa OpenCV: kembalikan format URL saja
                return url, user, pwd
    return None, None, None


def scan_single_ip(octet):
    """Pindai satu alamat IP (satu thread)."""
    ip = f"{CURRENT_SUBNET}{octet}"
    result = {"ip": ip, "ports": [], "rtsp_url": None, "user": None, "pwd": None}

    # Cek port RTSP (554)
    if check_port(ip, RTSP_PORT):
        result["ports"].append(554)

    # Cek port HTTP kamera (80 / 8080)
    if check_port(ip, ONVIF_PORT):
        result["ports"].append(80)
    if check_port(ip, HTTP_PORT):
        result["ports"].append(8080)

    if result["ports"]:
        return result
    return None


def print_banner():
    print("=" * 65)
    print("   CCTV IP SCANNER — Sistem Deteksi Jatuh BiLSTM")
    print("   Ellang Ramadhan | Politeknik Negeri Malang 2026")
    print("=" * 65)


def print_result(candidates):
    print("\n" + "=" * 65)
    print("  HASIL SCAN")
    print("=" * 65)
    if not candidates:
        print("\n  [!] Tidak ada perangkat dengan port RTSP/CCTV yang ditemukan.")
        print("\n  Kemungkinan penyebab:")
        print("  - Laptop dan CCTV tidak berada di jaringan WiFi yang sama")
        print("  - Port RTSP kamera diblokir oleh firewall router")
        print("  - Kamera menggunakan port non-standar (coba lihat di aplikasi CCTV)")
    else:
        print(f"\n  Ditemukan {len(candidates)} kandidat perangkat CCTV:\n")
        for i, cam in enumerate(candidates, 1):
            print(f"  [{i}] IP Address : {cam['ip']}")
            print(f"       Port terbuka : {cam['ports']}")
            if cam.get("rtsp_url"):
                print(f"       RTSP URL     : {cam['rtsp_url']}")
                print(f"       Username     : {cam['user']}")
                print(f"       Password     : {cam['pwd']}")
            else:
                print(f"       Saran RTSP URL:")
                for fmt in RTSP_FORMATS[:3]:
                    url = fmt.format(user="admin", pwd="admin", ip=cam["ip"])
                    print(f"         -> {url}")
            print()

        print("  Salin salah satu URL RTSP di atas ke kolom 'Camera URL'")
        print("  pada halaman web Dashboard Monitoring sistem Anda.")
    print("=" * 65)


# ─────────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()

    # Deteksi subnet otomatis
    CURRENT_SUBNET, local_ip = get_local_subnet()

    print(f"\n  IP Laptop Anda  : {local_ip}")
    print(f"  Subnet Target   : {CURRENT_SUBNET}0/24")
    print(f"  Port yang dicek : {RTSP_PORT} (RTSP), {ONVIF_PORT} (HTTP), {HTTP_PORT} (HTTP-Alt)")
    print(f"  Thread paralel  : {MAX_WORKERS}")

    # Izinkan override subnet manual
    answer = input(f"\n  Gunakan subnet {CURRENT_SUBNET} ? (Enter = Ya / ketik subnet lain, misal 10.0.0.): ").strip()
    if answer:
        CURRENT_SUBNET = answer if answer.endswith(".") else answer + "."
        print(f"  Menggunakan subnet: {CURRENT_SUBNET}0/24")

    print(f"\n  Memindai {CURRENT_SUBNET}1 - {CURRENT_SUBNET}254 ...")
    print("  Mohon tunggu...\n")

    start_time = time.time()
    candidates = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(scan_single_ip, range(1, 255)))

    for res in results:
        if res:
            print(f"  [+] Port terbuka ditemukan di: {res['ip']} -> Port: {res['ports']}")
            candidates.append(res)

    elapsed = time.time() - start_time
    print(f"\n  Scan selesai dalam {elapsed:.1f} detik.")

    # Opsional: coba koneksi RTSP otomatis
    if candidates:
        print(f"\n  Mencoba koneksi RTSP otomatis ke {len(candidates)} kandidat...")
        for cam in candidates:
            if 554 in cam["ports"]:
                print(f"    Menguji {cam['ip']}...", end=" ", flush=True)
                url, user, pwd = try_rtsp_connect(cam["ip"])
                if url:
                    cam["rtsp_url"] = url
                    cam["user"]     = user
                    cam["pwd"]      = pwd
                    print("BERHASIL!")
                else:
                    print("gagal (coba manual)")

    print_result(candidates)
    input("\n  Tekan Enter untuk keluar...")
