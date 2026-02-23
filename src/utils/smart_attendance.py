import cv2
import time
from deepface import DeepFace
import os

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_DB_PATH = os.path.join(BASE_DIR, 'datasets', 'faces')
os.makedirs(FACE_DB_PATH, exist_ok=True)

# Inisialisasi OpenCV Haar Cascades untuk Wajah & Mata (Bawaan OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

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
    """Fungsi untuk Clock-in Karyawan dengan Liveness (OpenCV) & Recognition"""
    registered_photo = os.path.join(FACE_DB_PATH, f"{employee_id}.jpg")
    
    if not os.path.exists(registered_photo):
        print(f"[ERR] Karyawan {employee_id} belum terdaftar!")
        return False

    cap = cv2.VideoCapture(0)
    print("Tatap kamera dan BERKEDIP untuk absen...")
    
    blinked = False
    frame_count = 0
    max_frames = 150 # Maksimal waktu tunggu (~5 detik)
    
    # Variabel untuk melacak status mata
    eyes_visible_previously = False

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. CEK LIVENESS DENGAN OPENCV (Apakah dia berkedip?)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Area wajah untuk mencari mata
            roi_gray = gray[y:y+h, x:x+w]
            
            # Deteksi mata di dalam wajah
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
            
            # Logika Kedipan: Jika sebelumnya mata terdeteksi, lalu sekarang tidak ada (tertutup)
            if len(eyes) > 0:
                eyes_visible_previously = True
            elif eyes_visible_previously and len(eyes) == 0:
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
            result = DeepFace.verify(
                img1_path=frame, 
                img2_path=registered_photo,
                model_name="Facenet", 
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
        print("❌ ABSEN DITOLAK! Liveness gagal (Tidak terdeteksi kedipan).")
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