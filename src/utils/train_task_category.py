import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'nlp')
DATA_DIR = os.path.join(BASE_DIR, 'datasets', 'synthetic')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 1. GENERATE DATASET (SYNTHETIC) ---
def generate_task_data():
    data = {
        "text": [],
        "category": []
    }
    
    # Kumpulan kata kunci untuk membuat kalimat random
    categories = {
        "DEVELOPMENT": [
            "implement api", "buat backend", "setup database", "refactor code", 
            "integrasi payment gateway", "create microservice", "coding fitur login",
            "update schema sql", "optimize query", "develop frontend dashboard"
        ],
        "BUGFIX": [
            "fix bug", "perbaiki error", "debug issue", "resolve crash", 
            "hotfix production", "patch security hole", "koreksi typo", 
            "handling exception", "troubleshoot server", "fix responsive layout"
        ],
        "MEETING": [
            "daily standup", "meeting klien", "diskusi requirement", "sprint planning",
            "brainstorming ide", "weekly sync", "retrospective", "presentasi progress",
            "call dengan vendor", "koordinasi tim"
        ],
        "DESIGN": [
            "buat mockup", "design ui/ux", "export aset gambar", "revisi layout",
            "pilih palet warna", "design logo", "wireframing", "prototyping figma",
            "update design system", "slicing design"
        ],
        "DEVOPS": [
            "deploy ke server", "setup ci/cd", "konfigurasi docker", "monitoring log",
            "backup database", "update sertifikat ssl", "manage aws instance",
            "skaling kubernetes", "cek penggunaan memori", "maintenance server"
        ]
    }

    # Generate variasi kalimat
    prefixes = ["tolong", "sedang", "progress", "selesai", "lanjut", ""]
    suffixes = ["hari ini", "secepatnya", "untuk klien", "di production", "pending", ""]

    print("[INFO] Generating synthetic NLP data...")
    for label, phrases in categories.items():
        for phrase in phrases:
            # Tambahkan kalimat asli
            data["text"].append(phrase)
            data["category"].append(label)
            
            # Tambahkan 5 variasi kalimat untuk setiap frase agar model lebih pintar
            for _ in range(5):
                p = np.random.choice(prefixes)
                s = np.random.choice(suffixes)
                sentence = f"{p} {phrase} {s}".strip()
                data["text"].append(sentence)
                data["category"].append(label)

    return pd.DataFrame(data)

# --- 2. TRAINING PIPELINE ---
def train_model():
    df = generate_task_data()
    
    # Simpan dataset untuk referensi
    df.to_csv(os.path.join(DATA_DIR, 'task_nlp_train.csv'), index=False)
    
    X = df['text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[INFO] Training on {len(df)} sentences...")
    
    # Pipeline: 
    # 1. TfidfVectorizer: Mengubah teks menjadi angka (vektor) berdasarkan frekuensi kata
    # 2. LogisticRegression: Mengklasifikasikan vektor tersebut ke kategori
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))), 
        # Ganti solver ke 'lbfgs' yang support multiclass native & tambah max_iter agar training tuntas
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, C=10))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluasi
    acc = pipeline.score(X_test, y_test)
    print(f"[RESULT] Model Accuracy: {acc*100:.2f}%")
    
    # Save Model
    model_path = os.path.join(MODEL_DIR, 'task_categorizer_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"[SUCCESS] Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()