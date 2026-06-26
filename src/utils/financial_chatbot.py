import pandas as pd
import json
import urllib.request
import urllib.error

# ==========================================
# 1. DATABASE LOKAL (Simulasi Data Finansial)
# ==========================================
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
# 2. LLAMA ENGINE (via Ollama)
# ==========================================
class FinancialChatbot:
    def __init__(self, df):
        self.df = df
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:7b"
        
        # Prepare context data as string for the LLM
        self.context_data = self._prepare_context()
        
    def _prepare_context(self):
        """Mengubah dataframe menjadi string teks yang bisa dipahami Llama."""
        context = "Berikut adalah data finansial proyek saat ini:\n\n"
        for _, row in self.df.iterrows():
            status = "Overbudget" if row['cost'] > row['budget'] else "Aman"
            context += (
                f"- Project {row['project_name']} (ID: {row['project_id']}):\n"
                f"  Budget: Rp {row['budget']:,}, Cost: Rp {row['cost']:,}, Status Budget: {status}\n"
                f"  Revenue: Rp {row['revenue']:,}, Margin: Rp {row['margin']:,} ({row['margin_pct']:.2f}%)\n"
            )
        
        # Total Overall
        tot_budget = self.df['budget'].sum()
        tot_cost = self.df['cost'].sum()
        tot_rev = self.df['revenue'].sum()
        tot_margin = self.df['margin'].sum()
        tot_pct = (tot_margin / tot_rev) * 100 if tot_rev > 0 else 0
        
        context += "\nOVERALL PERUSAHAAN:\n"
        context += f"Total Budget: Rp {tot_budget:,}, Total Cost: Rp {tot_cost:,}\n"
        context += f"Total Revenue: Rp {tot_rev:,}, Total Margin: Rp {tot_margin:,} ({tot_pct:.2f}%)\n"
        
        return context

    # ==========================================
    # 3. ENGINE UTAMA CHATBOT (Ollama API)
    # ==========================================
    def _build_prompt(self, context, user_input):
        return (
            "Anda adalah AI Asisten Finansial. Anda HANYA diizinkan menjawab pertanyaan yang relevan dengan data proyek, "
            "finansial, timesheet, atau pengelolaan sistem yang diberikan. JANGAN PERNAH menjawab hal-hal "
            "di luar konteks ini (contoh: pertanyaan tentang cuaca, memasak, presiden, atau topik umum lainnya). "
            "Jika pengguna bertanya hal di luar batas tersebut, tolaklah dengan sopan dan jelaskan peran Anda.\n\n"
            f"Konteks Data:\n{context}\n\n"
            f"Pertanyaan Pengguna: {user_input}\n"
            "Jawaban (Gunakan bahasa Indonesia, profesional, ringkas):"
        )

    def chat_with_context(self, user_input, custom_context):
        prompt = self._build_prompt(custom_context, user_input)
        return self._send_to_ollama(prompt)

    def chat(self, user_input):
        prompt = self._build_prompt(self.context_data, user_input)
        return self._send_to_ollama(prompt)

    def _send_to_ollama(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            req = urllib.request.Request(
                self.api_url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "Maaf, format respons dari model AI tidak dikenali.")
        except urllib.error.URLError as e:
            return f"Gagal menghubungi server Ollama. Pastikan Ollama berjalan di localhost:11434. Error: {e.reason}"
        except Exception as e:
            return f"Terjadi kesalahan: {str(e)}"

# ==========================================
# 4. SIMULASI TERMINAL CHAT
# ==========================================
if __name__ == "__main__":
    bot = FinancialChatbot(df_finance)
    print("🤖 Financial AI Chatbot (LLAMA 3 LOCAL) Siap!")
    print("Ketik 'exit' untuk keluar.\n")
    
    while True:
        teks = input("Anda: ")
        if teks.lower() == 'exit':
            break
        
        jawaban = bot.chat(teks)
        print(f"Bot : {jawaban}\n")