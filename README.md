# Capstone Master AI API - Dokumentasi Proyek

Aplikasi ini adalah platform AI All-in-One yang mengintegrasikan berbagai fitur cerdas mulai dari analisis keuangan, peramalan anggaran, hingga pengenalan wajah untuk absensi.

## 🚀 Fitur Utama

Aplikasi ini memiliki 6 modul utama:

1. **Task Categorizer (NLP)**: Mengklasifikasikan teks tugas ke dalam kategori tertentu.
2. **Task Complexity Recommender (NLP)**: Menganalisis judul dan deskripsi tugas untuk memberikan rekomendasi tingkat kerumitan (1-5).
3. **Timesheet Anomaly Detection**: Mendeteksi ketidakwajaran pada durasi kerja karyawan.
4. **Financial Chatbot**: Asisten interaktif untuk menanyakan status kesehatan keuangan, biaya, dan margin proyek.
5. **Budget Forecast**: Prediksi pengeluaran anggaran 90 hari ke depan menggunakan modul Prophet.
6. **Smart Attendance**: Sistem absensi berbasis pengenalan wajah (*Face Recognition*) menggunakan DeepFace.

---

## 📂 Struktur Folder

```text
Capstone-AI/
├── datasets/
│   ├── faces/              # Database foto karyawan (.jpg)
│   └── synthetic/          # Dataset CSV untuk finansial & event
├── models/
│   ├── anomaly/            # Model deteksi anomali (.pkl)
│   ├── forecasting/        # Model budget prophet (.json)
│   └── nlp/                # Model kategori tugas (.pkl)
├── src/
│   ├── core/
│   │   └── forecast_engine.py  # Logika peramalan anggaran
│   ├── scripts/
│   │   └── api.py              # Entry point FastAPI
│   └── utils/
│       ├── financial_chatbot.py      # Logika Chatbot Finansial
│       └── complexity_recommender.py # Logika Rekomendasi Kompleksitas Tugas
└── requirements.txt         # Daftar dependensi (perlu dibuat)

```

---

## 🛠 Langkah-langkah Persiapan

### 1. Prasyarat

Pastikan Anda sudah menginstal:

* Python 3.8 atau lebih baru.
* Library pendukung (FastAPI, Uvicorn, Joblib, Pandas, Prophet, DeepFace, OpenCV).

### 2. Instalasi Dependensi

Jalankan perintah berikut di terminal Anda:

```bash
pip install fastapi uvicorn joblib pandas numpy prophet deepface opencv-python matplotlib

```

---

## 🏃 Cara Menjalankan API

1. Buka terminal dan arahkan ke folder utama proyek (`Capstone-AI`).
2. Jalankan perintah berikut:
```bash
uvicorn src.scripts.api:app --reload

```


3. API akan berjalan di: `http://127.0.0.1:8000`
4. Buka dokumentasi interaktif (Swagger UI) di: `http://127.0.0.1:8000/docs`.

---

## 📖 Dokumentasi Endpoint API

### 1. Task Categorizer

* **Endpoint**: `POST /api/v1/task/categorize`
* **Input**: `{"task_text": "string"}`
* **Fungsi**: Memprediksi kategori dari teks tugas yang dimasukkan.

### 2. Task Complexity Recommender

* **Endpoint**: `POST /api/v1/task/complexity-recommend`
* **Input**: `{"title": "string", "description": "string"}`
* **Fungsi**: Menganalisis teks judul dan deskripsi tugas untuk memberikan rekomendasi skor kompleksitas (1-5) berdasarkan panjang teks dan penggunaan kata kunci teknis.

### 3. Check Anomaly

* **Endpoint**: `POST /api/v1/timesheet/check-anomaly`
* **Input**: JSON berisi `complexity`, `hist_avg`, `skill`, dan `duration`.
* **Output**: Status `SAFE` atau `SUSPICIOUS`.

### 4. Financial Chatbot

* **Endpoint**: `POST /api/v1/finance/chat`
* **Input**: `{"user_message": "Berapa pengeluaran project Alpha?"}`
* **Fungsi**: Menjawab pertanyaan seputar biaya, margin, atau kesehatan proyek.

### 5. Budget Forecast

* **Endpoint**: `GET /api/v1/finance/forecast/{project_id}`
* **Parameter**: `project_id` (contoh: `PROJ_ALPHA`, `PROJ_BETA`).
* **Fungsi**: Menghitung *Runway* (sisa waktu anggaran) dan prediksi biaya 30 hari ke depan.

### 6. Smart Attendance

* **Endpoint**: `POST /api/v1/attendance/verify`
* **Input**: `employee_id` (Form Data) dan `photo` (File Upload).
* **Fungsi**: Memverifikasi wajah karyawan dengan database foto di folder `datasets/faces/`.



---

## ⚠️ Catatan Penting

* **Model Files**: Pastikan file model `.pkl` dan `.json` tersedia di folder `models/` agar API tidak memberikan pesan *offline*.
* **Data Forecast**: Fitur peramalan membutuhkan file `multi_project_costs.csv` di folder `datasets/synthetic/`.
* **Smart Attendance**: Foto di folder `datasets/faces/` harus diberi nama sesuai `employee_id` (contoh: `EMP001.jpg`).
