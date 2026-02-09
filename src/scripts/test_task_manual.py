import joblib
import os
import pandas as pd

# Load Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'nlp', 'task_categorizer_model.pkl')

if not os.path.exists(MODEL_PATH):
    print("Model not found. Run 'src/utils/train_task_category.py' first.")
    exit()

model = joblib.load(MODEL_PATH)

def predict_task(task_text):
    # Prediksi Kategori
    category = model.predict([task_text])[0]
    
    # Ambil tingkat keyakinan (Confidence Score)
    probs = model.predict_proba([task_text])[0]
    max_prob = max(probs)
    
    # Logika UI: Beri ikon biar cantik
    icons = {
        "DEVELOPMENT": "💻", "BUGFIX": "🐞", "MEETING": "📅", 
        "DESIGN": "🎨", "DEVOPS": "🚀"
    }
    icon = icons.get(category, "❓")
    
    print(f"\nINPUT: '{task_text}'")
    print(f" -> AI Category : {icon} {category}")
    print(f" -> Confidence  : {max_prob*100:.1f}%")
    
    # Jika confidence rendah (< 50%), mungkin AI bingung
    if max_prob < 0.5:
        print("    (⚠️ AI ragu-ragu, mungkin perlu dikoreksi manusia)")

# --- TEST CASES ---
print("=== SMART TASK CATEGORIZER TEST ===")

# Skenario 1: Jelas
predict_task("benerin error di halaman login")
predict_task("meeting sama pak bos")
predict_task("slicing design dashboard admin")

# Skenario 2: Campuran Inggris/Indo (Model TF-IDF cukup kuat di sini)
predict_task("deploy to production server")
predict_task("fixing bug typo di navbar")

# Skenario 3: Kalimat agak ambigu
predict_task("diskusi soal api backend") # Harusnya Meeting atau Dev? Tergantung training
predict_task("makan siang") # Out of context test