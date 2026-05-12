from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys

# ==========================================
# 0. KONFIGURASI PATH & IMPORT MODULE LOKAL
# ==========================================
# Ambil path folder utama (Capstone-AI)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR) # Agar bisa import dari folder src/

# Inisialisasi Aplikasi FastAPI
app = FastAPI(title="Capstone Master AI API", version="1.0", description="API All-in-One untuk seluruh fitur AI Capstone")

# Konfigurasi CORS agar frontend HTML bisa terhubung
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua origin (untuk testing lokal)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD MODELS & MODULES (Aman dari Crash)
# ==========================================

# --- A. NLP Task Categorizer & Timesheet Anomaly ---
try:
    task_model = joblib.load(os.path.join(BASE_DIR, 'models', 'nlp', 'task_categorizer_model.pkl'))
    anomaly_model = joblib.load(os.path.join(BASE_DIR, 'models', 'anomaly', 'timesheet_model_latest.pkl'))
except Exception as e:
    task_model = None
    anomaly_model = None
    print(f"Warning: Model NLP/Anomaly gagal diload. {e}")

# --- B. Financial Chatbot ---
try:
    from src.utils.financial_chatbot import FinancialChatbot, df_finance
    bot_instance = FinancialChatbot(df_finance)
    chatbot_ready = True
except ImportError:
    chatbot_ready = False
    print("Warning: Modul Financial Chatbot tidak ditemukan.")

# --- C. Budget Forecast ---
try:
    from src.core.forecast_engine import run_analysis
    forecast_ready = True
except ImportError:
    forecast_ready = False
    print("Warning: Modul Prophet/Forecast gagal diload.")

# --- D. Smart Attendance (Face Recognition) ---
try:
    import cv2
    from deepface import DeepFace
    face_ready = True
except ImportError:
    face_ready = False
    print("Warning: OpenCV/DeepFace belum terinstal. Fitur Attendance dinonaktifkan.")


# ==========================================
# 2. STRUKTUR DATA INPUT (PYDANTIC)
# ==========================================
class TaskRequest(BaseModel):
    task_text: str

class TaskResponse(BaseModel):
    task_text: str
    category: str
    confidence: float

class AnomalyRequest(BaseModel):
    complexity: int
    hist_avg: float
    skill: int
    duration: float

class AnomalyResponse(BaseModel):
    status: str
    is_anomaly: bool

class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    reply: str

class AttendanceResponse(BaseModel):
    employee_id: str
    status: str
    accuracy: float


# ==========================================
# 3. ENDPOINTS API
# ==========================================

@app.get("/")
def read_root():
    return {
        "message": "Capstone Master AI API is Running!",
        "active_modules": {
            "Task_Categorizer": task_model is not None,
            "Timesheet_Anomaly": anomaly_model is not None,
            "Financial_Chatbot": chatbot_ready,
            "Budget_Forecast": forecast_ready,
            "Smart_Attendance": face_ready
        },
        "docs": "Kunjungi http://127.0.0.1:8000/docs untuk testing."
    }

# --- FITUR 1: TASK CATEGORIZER ---
@app.post("/api/v1/task/categorize", response_model=TaskResponse, tags=["Task & NLP"])
def predict_task(request: TaskRequest):
    if not task_model: raise HTTPException(status_code=500, detail="Task model is offline.")
    category = task_model.predict([request.task_text])[0]
    probs = task_model.predict_proba([request.task_text])[0]
    return {
        "task_text": request.task_text,
        "category": category,
        "confidence": float(max(probs) * 100)
    }

# --- FITUR 2: TIMESHEET ANOMALY ---
@app.post("/api/v1/timesheet/check-anomaly", response_model=AnomalyResponse, tags=["Task & NLP"])
def predict_anomaly(request: AnomalyRequest):
    if not anomaly_model: raise HTTPException(status_code=500, detail="Anomaly model is offline.")
    ratio = request.duration / request.hist_avg
    input_data = pd.DataFrame([[request.complexity, request.hist_avg, request.skill, request.duration, ratio]], 
                              columns=['complexity', 'hist_avg', 'skill', 'duration', 'deviation_ratio'])
    pred = anomaly_model.predict(input_data)[0]
    return {
        "status": "SAFE" if pred == 1 else "SUSPICIOUS",
        "is_anomaly": bool(pred == -1)
    }

# --- FITUR 3: FINANCIAL CHATBOT ---
@app.post("/api/v1/finance/chat", response_model=ChatResponse, tags=["Finance AI"])
def chat_finance(request: ChatRequest):
    if not chatbot_ready: raise HTTPException(status_code=500, detail="Chatbot module is offline.")
    jawaban = bot_instance.chat(request.user_message)
    return {"reply": jawaban}

# --- FITUR 4: BUDGET FORECAST ---
@app.get("/api/v1/finance/forecast/{project_id}", tags=["Finance AI"])
def get_forecast(project_id: str):
    """Contoh project_id: PROJ_ALPHA, PROJ_BETA, PROJ_GAMMA, PROJ_DELTA"""
    if not forecast_ready: raise HTTPException(status_code=500, detail="Forecast module is offline.")
    
    # Memanggil logika asli dari forecast_engine.py
    result = run_analysis(project_id.upper(), mode="PORTFOLIO")
    if result is None:
        raise HTTPException(status_code=404, detail="Data project tidak ditemukan atau model gagal.")
    
    return result

# --- FITUR 5: SMART ATTENDANCE (Upload Foto) ---
@app.post("/api/v1/attendance/verify", response_model=AttendanceResponse, tags=["Smart Attendance"])
async def verify_attendance(employee_id: str = Form(...), photo: UploadFile = File(...)):
    """Mengirim file gambar (selfie) untuk dicocokkan dengan database."""
    if not face_ready: raise HTTPException(status_code=500, detail="DeepFace is offline.")
    
    # 1. Cek apakah wajah karyawan ada di database (folder datasets/faces)
    db_photo_path = os.path.join(BASE_DIR, 'datasets', 'faces', f"{employee_id}.jpg")
    if not os.path.exists(db_photo_path):
        raise HTTPException(status_code=404, detail=f"Karyawan {employee_id} belum terdaftar.")

    # 2. Simpan foto upload sementara untuk DeepFace
    temp_path = f"temp_{employee_id}.jpg"
    with open(temp_path, "wb") as buffer:
        buffer.write(await photo.read())

    try:
        # 3. Lakukan pengenalan wajah
        result = DeepFace.verify(img1_path=temp_path, img2_path=db_photo_path, model_name="Facenet", enforce_detection=False)
        os.remove(temp_path) # Hapus foto temp
        
        return {
            "employee_id": employee_id,
            "status": "VERIFIED" if result["verified"] else "REJECTED",
            "accuracy": round(100 - (result['distance'] * 100), 2)
        }
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Face recognition error: {str(e)}")