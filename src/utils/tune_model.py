import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import os
import logging

# Matikan log sampah
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# --- KONFIGURASI ---
DATA_PATH = os.path.join('datasets', 'synthetic', 'project_cost.csv')
EVENTS_PATH = os.path.join('datasets', 'synthetic', 'project_events.csv')

def load_data():
    df = pd.read_csv(DATA_PATH)
    holidays = pd.read_csv(EVENTS_PATH) if os.path.exists(EVENTS_PATH) else None
    return df, holidays

def auto_tune():
    df, holidays = load_data()
    print(f"🔧 Memulai Auto-Tuning pada {len(df)} baris data...")

    # 1. Tentukan Grid Parameter (Kombinasi yang mau dicoba)
    # Kita akan mencoba semua kemungkinan di bawah ini:
    param_grid = {  
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5], # Sensitivitas Tren
        'seasonality_prior_scale': [0.1, 1.0, 10.0],       # Kekuatan Pola Musiman
        'seasonality_mode': ['additive', 'multiplicative'] # Jenis Musiman
    }

    # Generate semua kombinasi (4 x 3 x 2 = 24 Kombinasi)
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    results = []  

    # 2. Loop Semua Kombinasi
    print(f"Total Kombinasi yang akan dites: {len(all_params)}")
    
    for i, params in enumerate(all_params):
        print(f"[{i+1}/{len(all_params)}] Testing: {params} ...", end=" ")
        
        try:
            # Setup Model dengan parameter dinamis
            m = Prophet(**params, holidays=holidays, interval_width=0.95)
            m.add_country_holidays(country_name='ID')
            m.add_regressor('headcount')
            
            m.fit(df)

            # Ujian (Cross Validation)
            # Kita perpendek periode ujian biar tuningnya cepat (initial 200 hari)
            df_cv = cross_validation(m, initial='200 days', period='60 days', horizon='30 days', parallel="processes")
            
            # Hitung Error
            df_p = performance_metrics(df_cv)
            rmse = df_p['rmse'].mean()
            mape = df_p['mape'].mean() * 100

            results.append({
                **params,
                'rmse': rmse,
                'mape': mape
            })
            print(f"➡️ MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"❌ Gagal: {e}")

    # 3. Cari Pemenangnya
    tuning_results = pd.DataFrame(results)
    best_params = tuning_results.sort_values('mape').iloc[0]

    print("\n" + "="*50)
    print("🏆 JUARA 1 (SETTINGAN TERBAIK)")
    print("="*50)
    print(f"MAPE Terendah : {best_params['mape']:.2f}%")
    print(f"RMSE Terendah : {best_params['rmse']:.0f}")
    print("-" * 30)
    print("👇 COPY PARAMETER INI KE forecast_engine.py 👇")
    print(f"changepoint_prior_scale = {best_params['changepoint_prior_scale']}")
    print(f"seasonality_prior_scale = {best_params['seasonality_prior_scale']}")
    print(f"seasonality_mode        = '{best_params['seasonality_mode']}'")
    print("="*50)

if __name__ == "__main__":
    auto_tune() 