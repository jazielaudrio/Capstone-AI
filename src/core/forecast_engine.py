import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import os
import logging

# --- CONFIG ---
DEV_MODE = True
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Config disamakan dengan Generator
PROJECT_CONFIGS = {
    "PROJ_ALPHA": { "name": "Alpha (Enterprise)", "budget": 80000000000 },
    "PROJ_BETA":  { "name": "Beta (Growth)", "budget": 5000000000 },
    "PROJ_GAMMA": { "name": "Gamma (Declining)", "budget": 3000000000 },
    "PROJ_DELTA": { "name": "Delta (Volatile)", "budget": 10000000000 }
}

BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'synthetic', 'multi_project_costs.csv')
EVENTS_PATH = os.path.join(BASE_DIR, 'datasets', 'synthetic', 'multi_project_events.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'forecasting', 'budget')

def get_data(pid):
    if not os.path.exists(DATA_PATH): raise FileNotFoundError("[ERR] Dataset not found.")
    df = pd.read_csv(DATA_PATH)
    df_proj = df[df['project_id'] == pid].copy()
    
    holidays = None
    if os.path.exists(EVENTS_PATH):
        ev = pd.read_csv(EVENTS_PATH)
        holidays = ev[ev['project_id'] == pid].copy()
        
    if df_proj.empty: raise ValueError(f"[ERR] No data for {pid}")
    return df_proj, holidays

def train(df, holidays):
    if DEV_MODE: print(f"[INFO] Training model on {len(df)} records...")
    
    # [F1] Linear growth for daily cost stability
    # [F3] Risk Guard 95% interval
    # [F4] Shock Absorber via holidays
    model = Prophet(
        growth='linear', interval_width=0.95, holidays=holidays,
        changepoint_prior_scale=0.05, seasonality_mode='multiplicative',
        daily_seasonality=False
    )
    model.add_country_holidays(country_name='ID') # [F2] Smart Calendar
    model.add_regressor('headcount')              # [F6] Scenario Planning
    model.fit(df)
    return model

def save_model(model, pid, mape):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"model_{pid}.json")
    
    # [F7] Quality Gate
    status = "PASSED" if mape < 20.0 else "WARNING"
    print(f"[QC] Model Quality: {status} (Error: {mape:.2f}%)")
    
    with open(path, 'w') as f: f.write(model_to_json(model))
    return path

def explain_forecast(forecast):
    """[F10] Explainable AI Logic"""
    fut = forecast.tail(30)
    trend = fut['trend'].mean()
    seasonal = fut['multiplicative_terms'].mean()
    
    narrative = f"Baseline Trend: IDR {trend:,.0f}/day."
    if abs(seasonal) > 0.01:
        impact = "increases" if seasonal > 0 else "reduces"
        narrative += f" Seasonality {impact} cost by {abs(seasonal)*100:.1f}%."
    return narrative

def run_analysis(pid, mode="SINGLE"):
    try:
        cfg = PROJECT_CONFIGS[pid]
        df, holidays = get_data(pid)
        model = train(df, holidays)
        
        # Eval
        cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days', parallel="processes")
        mape = performance_metrics(cv)['mape'].mean() * 100
        save_model(model, pid, mape)
        
        # Forecast
        future = model.make_future_dataframe(periods=90)
        future['headcount'] = df['headcount'].iloc[-1]
        forecast = model.predict(future)
        
        # [F8] Runway Calculation
        spent = df['y'].sum()
        budget = cfg['budget']
        
        if spent >= budget:
            status = "CRITICAL_OVER"
            runway = None
        else:
            future_fc = forecast[forecast['ds'] > df['ds'].max()].copy()
            future_fc['cumsum'] = future_fc['yhat'].cumsum() + spent
            over = future_fc[future_fc['cumsum'] >= budget]
            runway = over.iloc[0]['ds'] if not over.empty else None
            status = "WARNING" if runway else "SAFE"

        # [F9] Portfolio Data Preparation
        next_month = forecast.tail(30)['yhat'].sum()
        explanation = explain_forecast(forecast)
        
        result = {
            "project": cfg['name'],
            "budget": budget,
            "spent": spent,
            "pct": (spent/budget)*100,
            "status": status,
            "runway": runway,
            "forecast_30d": next_month,
            "explanation": explanation
        }

        if mode == "SINGLE":
            print_report(result)
            if DEV_MODE:
                model.plot(forecast)
                plt.title(f"{cfg['name']} (MAPE: {mape:.2f}%)")
                plt.show()
                
        return result

    except Exception as e:
        print(f"[ERR] {e}")
        return None

def print_report(res):
    print("\n" + "-"*60)
    print(f"REPORT: {res['project']}")
    print("-"*60)
    print(f"FINANCIAL HEALTH:")
    print(f"  Budget     : IDR {res['budget']:,.0f}")
    print(f"  Actual     : IDR {res['spent']:,.0f} ({res['pct']:.1f}%)")
    print(f"  Status     : {res['status']}")
    
    if res['runway']:
        print(f"  Runway End : {res['runway'].strftime('%Y-%m-%d')}")
    elif res['status'] == "CRITICAL_OVER":
        print(f"  Runway End : ALREADY EXCEEDED")
    else:
        print(f"  Runway End : Safe > 90 Days")
        
    print(f"\nFORECAST INSIGHT:")
    print(f"  Next 30 Days : IDR {res['forecast_30d']:,.0f}")
    print(f"  AI Logic     : {res['explanation']}")
    print("-"*60 + "\n")

if __name__ == "__main__":
    # --- SELECT MODE ---
    # Options: SINGLE (Detail + Graph) or PORTFOLIO (Summary Table)
    EXEC_MODE = "SINGLE"
    
    if EXEC_MODE == "PORTFOLIO":
        print("[INFO] Starting Portfolio Analysis...\n")
        agg_forecast = 0
        for pid in PROJECT_CONFIGS:
            r = run_analysis(pid, mode="PORTFOLIO")
            if r: 
                agg_forecast += r['forecast_30d']
                print(f"{r['project']:<30} | {r['status']:<15} | Fcst: {r['forecast_30d']:,.0f}")
        
        print("\n" + "="*60)
        print(f"TOTAL COMPANY CASHFLOW NEEDED (Next 30 Days): IDR {agg_forecast:,.0f}")
        print("="*60)

    else:
        # Test specific volatile project
        run_analysis("PROJ_DELTA", mode="SINGLE")