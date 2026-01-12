import pandas as pd
import numpy as np
import joblib
import os
import datetime
import shutil
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# ==========================================
# 1. SETUP PATH & CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'anomaly')
DATA_DIR = os.path.join(BASE_DIR, 'datasets', 'synthetic')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# KONFIGURASI UTAMA
# Kita turunkan threshold karena data "Smart Cheater" lebih sulit diprediksi
MIN_ACCURACY_THRESHOLD = 0.85  
TOTAL_SAMPLES = 5000           
CONTAMINATION_RATE = 0.05      

print(f"[INFO] Config: Samples={TOTAL_SAMPLES}, Contamination={CONTAMINATION_RATE}, Min Acc={MIN_ACCURACY_THRESHOLD}")

# ==========================================
# 2. GENERATE DATA (SMART CHEATER EDITION)
# ==========================================
np.random.seed(42)

def generate_data(n_samples=1000, anomaly=False):
    data = []
    for _ in range(n_samples):
        complexity = np.random.randint(1, 6) # 1-5
        skill = np.random.randint(1, 4)      # 1-3
        hist_avg = complexity * 2.0          
        
        if not anomaly:
            # === LOGIKA NORMAL (Toleransi Junior) ===
            # Junior (1): Noise lebar (0.8 - 1.6) -> Boleh lambat
            if skill == 1:
                skill_factor = 1.3
                noise = np.random.uniform(0.8, 1.6) 
            elif skill == 2:
                skill_factor = 1.0
                noise = np.random.uniform(0.8, 1.2)
            else:
                skill_factor = 0.8
                noise = np.random.uniform(0.7, 1.1)
                
            duration = hist_avg * skill_factor * noise
            label = 0 # Normal
        else:
            # === LOGIKA ANOMALI (SMART CHEATER) ===
            # Markup tipis (1.6x - 2.5x) agar mirip dengan Junior yg lambat
            # Ini membuat model bekerja keras membedakan pola.
            markup_factor = np.random.uniform(1.6, 2.5)
            duration = hist_avg * markup_factor
            label = 1 # Anomaly
            
        data.append([complexity, hist_avg, skill, duration, label])
        
    return pd.DataFrame(data, columns=['complexity', 'hist_avg', 'skill', 'duration', 'label'])

# Generate Data: 95% Normal, 5% Anomaly
print(f"[INFO] Generating {TOTAL_SAMPLES} synthetic records (Realistis)...")
n_normal = int(TOTAL_SAMPLES * (1 - CONTAMINATION_RATE))
n_anomaly = int(TOTAL_SAMPLES * CONTAMINATION_RATE)

df = pd.concat([generate_data(n_normal, False), generate_data(n_anomaly, True)], ignore_index=True)

# Feature Engineering
df['deviation_ratio'] = df['duration'] / df['hist_avg']

# Save CSV untuk audit manual
csv_path = os.path.join(DATA_DIR, 'timesheet_anomaly_train_data.csv')
df.to_csv(csv_path, index=False)
print(f"[INFO] Dataset saved to CSV for review.")

# ==========================================
# 3. TRAINING
# ==========================================
features = ['complexity', 'hist_avg', 'skill', 'duration', 'deviation_ratio']
X = df[features]
y_true = df['label'] 

print("[INFO] Training Isolation Forest...")
model = IsolationForest(contamination=CONTAMINATION_RATE, random_state=42, n_jobs=-1)
model.fit(X)

# ==========================================
# 4. EVALUATION
# ==========================================
# Convert prediksi model (-1/1) ke format label kita (1/0)
preds_raw = model.predict(X)
preds_converted = [1 if x == -1 else 0 for x in preds_raw]

accuracy = accuracy_score(y_true, preds_converted)
acc_percent = round(accuracy * 100, 2)

print("-" * 30)
print(f"MODEL ACCURACY: {acc_percent}%")
print("-" * 30)

# ==========================================
# 5. SAVE MODEL (WITH BACKUP & CHECK)
# ==========================================
MODEL_PATH = os.path.join(MODEL_DIR, 'timesheet_model.pkl')
LATEST_PATH = os.path.join(MODEL_DIR, 'timesheet_model_latest.pkl')

if accuracy >= MIN_ACCURACY_THRESHOLD:
    # 1. Backup model lama jika ada
    if os.path.exists(LATEST_PATH):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"timesheet_model_BACKUP_{timestamp}.pkl"
        shutil.copy(LATEST_PATH, os.path.join(MODEL_DIR, backup_name))
        print(f"[BACKUP] Old model backed up to {backup_name}")

    # 2. Simpan model baru (Versi Arsip)
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"timesheet_model_{today_str}_ACC{acc_percent}.pkl"
    save_path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, save_path)
    
    # 3. Simpan model baru (Versi Production/Latest)
    joblib.dump(model, LATEST_PATH)
    
    print(f"[SUCCESS] Model Saved! Accuracy ({acc_percent}%) passed threshold.")
    print(f"[PATH] {LATEST_PATH}")
else:
    print(f"[FAILED] Accuracy ({acc_percent}%) is below threshold ({MIN_ACCURACY_THRESHOLD*100}%). Model NOT saved.")