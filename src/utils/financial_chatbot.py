import pandas as pd
import re

# ==========================================
# 1. DATABASE LOKAL (Simulasi Data Finansial)
# ==========================================
# Dalam implementasi nyata, data ini diambil dari SQL / CSV internal perusahaan
data = {
    "project_id": ["PROJ_ALPHA", "PROJ_BETA", "PROJ_GAMMA"],
    "project_name": ["Alpha", "Beta", "Gamma"],
    "budget": [80000000, 50000000, 30000000],
    "cost": [65000000, 20000000, 35000000],   # Actual spend
    "revenue": [100000000, 80000000, 25000000] # Pendapatan
}
df_finance = pd.DataFrame(data)

# Hitung Margin (Revenue - Cost) secara dinamis
df_finance["margin"] = df_finance["revenue"] - df_finance["cost"]
df_finance["margin_pct"] = (df_finance["margin"] / df_finance["revenue"]) * 100

# ==========================================
# 2. NLP ENGINE (Pendeteksi Niat & Entitas)
# ==========================================
class FinancialChatbot:
    def __init__(self, df):
        self.df = df
        
        # Kamus Niat (Intents)
        self.intents = {
            "health": [r"kesehatan", r"status", r"kondisi", r"aman", r"overbudget"],
            "cost": [r"cost", r"biaya", r"pengeluaran", r"habis berapa", r"spent"],
            "margin": [r"revenue", r"margin", r"pendapatan", r"profit", r"untung", r"keuntungan"]
        }
        
    def extract_entity(self, text):
        """Mencari nama project yang disebutkan dalam kalimat."""
        text = text.lower()
        if "semua" in text or "overall" in text or "total" in text:
            return "OVERALL"
            
        for name in self.df['project_name'].str.lower():
            if name in text:
                return name.capitalize()
        return None # Jika tidak ada project spesifik yang disebut

    def detect_intent(self, text):
        """Mendeteksi apa yang ingin ditanyakan user."""
        text = text.lower()
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        return "unknown"

    # ==========================================
# 3. LOGIKA JAWABAN (Berdasarkan Fungsi yang Diminta)
    # ==========================================
    def get_health(self, project):
        if project == "OVERALL":
            return "Untuk melihat kesehatan, sebutkan nama project spesifik (contoh: status project Alpha)."
            
        proj_data = self.df[self.df['project_name'] == project].iloc[0]
        budget = proj_data['budget']
        cost = proj_data['cost']
        pct_used = (cost / budget) * 100
        
        status = "🔴 BAHAYA (Overbudget)" if cost > budget else "🟢 AMAN"
        if 80 <= pct_used <= 100: status = "🟡 PERINGATAN (Hampir Habis)"
            
        return f"Kesehatan Project {project}: {status}\nBudget terpakai: {pct_used:.1f}% (Rp {cost:,} dari Rp {budget:,})"

    def get_cost(self, project):
        if project == "OVERALL":
            total_cost = self.df['cost'].sum()
            return f"Total pengeluaran (cost) untuk seluruh project adalah: Rp {total_cost:,}"
            
        cost = self.df[self.df['project_name'] == project].iloc[0]['cost']
        return f"Total pengeluaran (cost) untuk Project {project} saat ini adalah: Rp {cost:,}"

    def get_margin(self, project):
        if project == "OVERALL":
            tot_rev = self.df['revenue'].sum()
            tot_margin = self.df['margin'].sum()
            tot_pct = (tot_margin / tot_rev) * 100 if tot_rev > 0 else 0
            return (f"📊 OVERALL PERUSAHAAN:\n"
                    f"Total Revenue : Rp {tot_rev:,}\n"
                    f"Total Margin  : Rp {tot_margin:,}\n"
                    f"Margin (%)    : {tot_pct:.2f}%")
            
        proj_data = self.df[self.df['project_name'] == project].iloc[0]
        return (f"📊 Data Project {project}:\n"
                f"Revenue : Rp {proj_data['revenue']:,}\n"
                f"Margin  : Rp {proj_data['margin']:,} ({proj_data['margin_pct']:.2f}%)")

    # ==========================================
# 4. ENGINE UTAMA CHATBOT
    # ==========================================
    def chat(self, user_input):
        intent = self.detect_intent(user_input)
        entity = self.extract_entity(user_input)
        
        if not entity and intent != "unknown":
            return "Tolong sebutkan nama projectnya atau ketik 'overall' (Contoh: 'Berapa margin project Alpha?')"
            
        if intent == "health":
            return self.get_health(entity)
        elif intent == "cost":
            return self.get_cost(entity)
        elif intent == "margin":
            return self.get_margin(entity)
        else:
            return "Maaf, saya tidak mengerti. Anda bisa bertanya tentang 'status kesehatan', 'pengeluaran/cost', atau 'revenue/margin' dari sebuah project."

# ==========================================
# 5. SIMULASI TERMINAL CHAT
# ==========================================
if __name__ == "__main__":
    bot = FinancialChatbot(df_finance)
    print("🤖 Financial AI Chatbot (LOCAL) Siap!")
    print("Ketik 'exit' untuk keluar.\n")
    
    while True:
        teks = input("Anda: ")
        if teks.lower() == 'exit':
            break
        
        jawaban = bot.chat(teks)
        print(f"Bot : {jawaban}\n")