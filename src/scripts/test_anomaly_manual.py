import joblib
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. LOAD MODEL
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'anomaly', 'timesheet_model_latest.pkl')

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}. Run training script first.")
    exit()

print(f"[INFO] Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ==========================================
# 2. FUNGSI PREDIKSI
# ==========================================
def check_anomaly(case_name, complexity, hist_avg, skill, duration):
    # Hitung rasio
    ratio = duration / hist_avg
    
    # Siapkan data frame (format harus sama persis dengan saat training)
    input_data = pd.DataFrame([[complexity, hist_avg, skill, duration, ratio]], 
                              columns=['complexity', 'hist_avg', 'skill', 'duration', 'deviation_ratio'])
    
    # Predict (1 = Normal, -1 = Anomaly)
    pred = model.predict(input_data)[0]
    score = model.decision_function(input_data)[0]
    
    # Logika Status
    status = "✅ SAFE" if pred == 1 else "🚨 SUSPICIOUS"
    
    print("-" * 50)
    print(f"TEST CASE: {case_name}")
    print(f"   -> Detail: Level {skill} | Task Avg {hist_avg}h | Input {duration}h")
    print(f"   -> Rasio : {ratio:.2f}x lipat")
    print(f"   -> HASIL : {status} (Score: {score:.4f})")

# ==========================================
# 3. JALANKAN SKENARIO UJI (STRESS TEST)
# ==========================================
print("\n=== STARTING MODEL VERIFICATION ===\n")

# KASUS 1: Normal Worker (Mid Level)
# Kerja sesuai rata-rata. Harusnya AMAN.
check_anomaly("The Good Worker", 
              complexity=3, hist_avg=6.0, skill=2, duration=6.1)

# KASUS 2: The 'Slow' Junior (CRITICAL TEST)
# Junior (Skill 1) kerja 1.5x lebih lama.
# DI MODEL LAMA: Ini kena "Suspicious".
# DI MODEL BARU: Ini harusnya "SAFE" (karena kita sudah latih toleransi Junior).
check_anomaly("Junior Lambat (Jujur)", 
              complexity=2, hist_avg=4.0, skill=1, duration=6.0)

# KASUS 3: The 'Smart' Cheater (CRITICAL TEST)
# Senior (Skill 3) mencoba mark-up tipis (1.8x).
# Dia pura-pura lambat, padahal senior harusnya cepat.
# Harusnya TERDETEKSI.
check_anomaly("Pencuri Pintar (Mark-up Tipis)", 
              complexity=2, hist_avg=4.0, skill=3, duration=7.2)

# KASUS 4: Brutal Cheater
# Mark-up gila-gilaan (4x).
# Harusnya PASTI TERDETEKSI.
check_anomaly("Pencuri Rakus", 
              complexity=1, hist_avg=2.0, skill=2, duration=8.0)