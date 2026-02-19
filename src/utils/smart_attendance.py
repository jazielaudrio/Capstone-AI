import cv2
import mediapipe as mp
import time
from deepface import DeepFace
import os

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_DB_PATH = os.path.join(BASE_DIR, 'datasets', 'faces')
os.makedirs(FACE_DB_PATH, exist_ok=True)

# Inisialisasi MediaPipe untuk deteksi wajah & mata (Liveness)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Indeks titik mata pada MediaPipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(eye_points, landmarks):
    """Menghitung Eye Aspect Ratio (EAR) untuk deteksi kedipan"""
    # Secara sederhana: Jarak vertikal mata dibagi jarak horizontal mata
    # Jika EAR turun drastis, berarti mata sedang tertutup (berkedip)
    # (Kode disederhanakan untuk keperluan prototipe)
    
    # Ambil koordinat Y dari kelopak mata atas dan bawah
    upper_y = landmarks[eye_points[1]].y
    lower_y = landmarks[eye_points[4]].y
    
    # Hitung jarak (jika jarak sangat kecil, mata tertutup)
    distance = abs(lower_y - upper_y)
    return distance

def register_face(employee_id):
    """Fungsi untuk menyimpan wajah saat pertama kali masuk (Admin)"""
    cap = cv2.VideoCapture(0)
    print("Silakan lihat ke kamera. Menangkap wajah dalam 3 detik...")
    time.sleep(3)
    ret, frame = cap.read()
    if ret:
        path = os.path.join(FACE_DB_PATH, f"{employee_id}.jpg")
        cv2.imwrite(path, frame)
        print(f"[SUCCESS] Wajah terdaftar untuk ID: {employee_id}")
    cap.release()

def clock_in_attendance(employee_id):
    """Fungsi untuk Clock-in Karyawan dengan Liveness & Recognition"""
    registered_photo = os.path.join(FACE_DB_PATH, f"{employee_id}.jpg")
    
    if not os.path.exists(registered_photo):
        print(f"[ERR] Karyawan {employee_id} belum terdaftar!")
        return False

    cap = cv2.VideoCapture(0)
    print("Tatap kamera dan BERKEDIP untuk absen...")
    
    blinked = False
    blink_threshold = 0.015 # Batas jarak mata tertutup (sesuaikan saat tes)
    frame_count = 0
    max_frames = 150 # Maksimal waktu tunggu (~5 detik)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. CEK LIVENESS (Apakah dia berkedip?)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                left_ear = calculate_ear(LEFT_EYE, landmarks)
                right_ear = calculate_ear(RIGHT_EYE, landmarks)
                
                # Jika jarak kelopak mata sangat kecil -> Dia berkedip!
                if left_ear < blink_threshold and right_ear < blink_threshold:
                    blinked = True
                    print("[INFO] Liveness LULUS (Kedipan Terdeteksi)!")
                    break
        
        # Tampilkan ke layar
        cv2.putText(frame, "Tatap layar dan Berkedip", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Smart Attendance", frame)
        
        if blinked:
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1

    # 2. JIKA LIVENESS LULUS, LAKUKAN FACE RECOGNITION
    if blinked:
        print("[INFO] Memverifikasi Identitas...")
        try:
            # Bandingkan frame saat ini dengan foto di database
            # Enforce detection = False agar tidak error jika wajah sedikit terpotong
            result = DeepFace.verify(
                img1_path=frame, 
                img2_path=registered_photo,
                model_name="Facenet", # Model ringan dan akurat
                enforce_detection=False 
            )
            
            if result["verified"]:
                print(f"✅ ABSEN BERHASIL! Selamat bekerja, {employee_id}.")
                print(f"   -> Akurasi kemiripan: {100 - (result['distance']*100):.2f}%")
                status = True
            else:
                print("❌ ABSEN DITOLAK! Wajah tidak cocok dengan database.")
                status = False
                
        except Exception as e:
            print(f"[ERR] Gagal mendeteksi wajah dengan jelas: {e}")
            status = False
    else:
        print("❌ ABSEN DITOLAK! Liveness gagal (Tidak terdeteksi kehidupan/kedipan).")
        status = False

    cap.release()
    cv2.destroyAllWindows()
    return status

# --- CARA TESTING ---
if __name__ == "__main__":
    # Skenario 1: Daftarkan wajah dulu (Jalankan ini sekali saja)
    # register_face("EMP_001")
    
    # Skenario 2: Karyawan mencoba Absen
    print("\n--- MULAI CLOCK IN ---")
    clock_in_attendance("EMP_001") 