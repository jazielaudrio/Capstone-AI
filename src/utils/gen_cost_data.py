import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# --- KONFIGURASI 4 TIPE PROJECT BERBEDA ---
PROJECTS_DB = [
    {
        "id": "PROJ_ALPHA",
        "name": "Alpha (Enterprise/Stable)",
        "budget": 80000000000, 
        "start_hc": 50, "max_hc": 55, # Headcount stabil
        "rate": 1500000, "volatility": 0.05, "shock_prob": 0.2
    },
    {
        "id": "PROJ_BETA",
        "name": "Beta (Startup/High Growth)",
        "budget": 5000000000, 
        "start_hc": 5, "max_hc": 25, # Headcount naik drastis
        "rate": 1000000, "volatility": 0.1, "shock_prob": 0.1
    },
    {
        "id": "PROJ_GAMMA",
        "name": "Gamma (Maintenance/Declining)",
        "budget": 3000000000, 
        "start_hc": 10, "max_hc": 10, # Headcount turun (fase closing)
        "rate": 1200000, "volatility": 0.03, "shock_prob": 0.05
    },
    {
        "id": "PROJ_DELTA",
        "name": "Delta (Crisis/Volatile)",
        "budget": 10000000000, 
        "start_hc": 15, "max_hc": 20,
        "rate": 1300000, "volatility": 0.3, # Sangat fluktuatif (Bahaya)
        "shock_prob": 0.5 # Sering di-hold
    }
]

YEARS = 3
DAYS = 365 * YEARS
START_DATE = datetime.now() - timedelta(days=DAYS)

def get_seasonality(date):
    # [F2] Simulasi pola mingguan & akhir bulan
    if date.weekday() >= 5: return 0.9 # Weekend cost 90% (Lembur/Server)
    if date.day >= 25: return 1.2      # End month spike
    return 1.0

def generate_data():
    print(f"[INFO] Generating dataset: {YEARS} Years ({DAYS} rows/project)...")
    
    costs, events = [], []

    for p in PROJECTS_DB:
        print(f"[INFO] Processing: {p['name']}")
        curr_hc = p['start_hc']
        shock_rem = 0
        is_shock = False

        for i in range(DAYS):
            date = START_DATE + timedelta(days=i)
            
            # [F6] Headcount Dynamics
            if i % 30 == 0 and i > 0:
                # Logic beda tiap tipe project
                if "Growth" in p['name']: change = random.choice([0, 1, 2])
                elif "Declining" in p['name']: change = random.choice([-1, 0])
                else: change = random.choice([-1, 0, 1])
                
                curr_hc = max(2, min(p['max_hc'], curr_hc + change))

            # [F4] Shock Injection (Project Hold)
            if (not is_shock and random.random() < 0.001) or (i == 300):
                if random.random() < p['shock_prob']:
                    is_shock = True
                    shock_rem = random.randint(5, 14)
                    events.append({
                        'project_id': p['id'], 'holiday': 'project_hold',
                        'ds': date, 'lower_window': 0, 'upper_window': shock_rem
                    })

            # Base Cost Calculation
            base = curr_hc * p['rate']
            season = get_seasonality(date)
            noise = np.random.normal(0, base * p['volatility'])
            daily = (base * season) + noise
            
            if is_shock:
                daily = base * 0.1 # Maintenance only
                shock_rem -= 1
                if shock_rem <= 0: is_shock = False

            costs.append({
                'project_id': p['id'],
                'ds': date,
                'y': max(0, round(daily)),
                'cap': p['budget'],    # [F1] Data Plafon
                'headcount': curr_hc   # [F6] Data Regressor
            })

    # Export
    os.makedirs('datasets/synthetic', exist_ok=True)
    pd.DataFrame(costs).to_csv('datasets/synthetic/multi_project_costs.csv', index=False)
    pd.DataFrame(events).to_csv('datasets/synthetic/multi_project_events.csv', index=False)
    print("[SUCCESS] Data generated in datasets/synthetic/")

if __name__ == "__main__":
    generate_data()