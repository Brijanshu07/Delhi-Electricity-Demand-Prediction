import json
import os
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

LAT_DELHI = 28.6128
LON_DELHI = 77.2311
TOMORROW_API_KEY = os.environ.get("TOMORROW_API_KEY", "")
TOMORROW_FORECAST_URL = "https://api.tomorrow.io/v4/weather/forecast"

CALIBRATION_FACTOR = 0.85

import google.generativeai as genai
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR.parent / "MODEL"
PROCESSED_DIR = MODEL_DIR / "processed"

model = None
feature_columns = []
model_metadata = {}

def load_model():
    global model, feature_columns, model_metadata
    model_path = PROCESSED_DIR / "demand_model.joblib"
    if not model_path.exists():
        return False
    model = joblib.load(model_path)
    with open(PROCESSED_DIR / "feature_list.json") as f:
        feature_columns = json.load(f)
    meta_path = PROCESSED_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            model_metadata = json.load(f)
    return True

app = FastAPI(title="Delhi Electricity Demand API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://6989d2db8fe20a0008562eb8--delhi-peak-load-prediciton.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    date: str
    scenario: str
    temp: float
    humidity: float

class PredictResponse(BaseModel):
    predicted_load: float
    confidence: float

def build_features_for_datetime(dt: pd.Timestamp, temp: float = 30.0, rhum: float = 50.0) -> dict:
    hour = dt.hour if hasattr(dt, "hour") else pd.Timestamp(dt).hour
    month = dt.month if hasattr(dt, "month") else pd.Timestamp(dt).month
    day = dt.day if hasattr(dt, "day") else pd.Timestamp(dt).day
    day_of_week = dt.dayofweek if hasattr(dt, "dayofweek") else pd.Timestamp(dt).dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)
    rhum = float(rhum)
    dwpt = temp - (100 - rhum) / 5
    wdir, wspd, pres = 180.0, 3.0, 1010.0
    cooling_degree = max(0.0, temp - 24)
    temp_x_rhum = temp * (rhum / 100.0)
    peak_hour = 1 if 10 <= hour <= 18 else 0
    summer_month = 1 if month in (4, 5, 6) else 0
    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "temp": temp,
        "rhum": rhum,
        "dwpt": dwpt,
        "wdir": wdir,
        "wspd": wspd,
        "pres": pres,
        "month": month,
        "day": day,
        "sin_month": sin_month,
        "cos_month": cos_month,
        "cooling_degree": cooling_degree,
        "temp_x_rhum": temp_x_rhum,
        "peak_hour": peak_hour,
        "summer_month": summer_month,
    }

def build_features_from_scenario(date_str: str, temp: float, humidity: float, scenario: str) -> dict:
    dt = pd.to_datetime(date_str)
    hour = dt.hour
    month = dt.month
    day = dt.day
    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)
    rhum = float(humidity)
    dwpt = temp - (100 - rhum) / 5
    wdir, wspd, pres = 180.0, 3.0, 1010.0
    if scenario == "high-temp":
        temp = min(50, temp + 3)
        dwpt = temp - (100 - rhum) / 5
    cooling_degree = max(0.0, temp - 24)
    temp_x_rhum = temp * (rhum / 100.0)
    peak_hour = 1 if 10 <= hour <= 18 else 0
    summer_month = 1 if month in (4, 5, 6) else 0
    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "temp": temp,
        "rhum": rhum,
        "dwpt": dwpt,
        "wdir": wdir,
        "wspd": wspd,
        "pres": pres,
        "month": month,
        "day": day,
        "sin_month": sin_month,
        "cos_month": cos_month,
        "cooling_degree": cooling_degree,
        "temp_x_rhum": temp_x_rhum,
        "peak_hour": peak_hour,
        "summer_month": summer_month,
    }

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    df = df.copy()
    if "month" not in df.columns and df["datetime"].dtype != object:
        df["month"] = pd.to_datetime(df["datetime"]).dt.month
    if "day" not in df.columns and df["datetime"].dtype != object:
        df["day"] = pd.to_datetime(df["datetime"]).dt.day
    if "hour" not in df.columns and df["datetime"].dtype != object:
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    dt = pd.to_datetime(df["datetime"])
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["cooling_degree"] = (df["temp"] - 24).clip(lower=0)
    df["temp_x_rhum"] = df["temp"] * (df["rhum"] / 100.0)
    df["peak_hour"] = ((df["hour"] >= 10) & (df["hour"] <= 18)).astype(int)
    df["summer_month"] = df["month"].isin([4, 5, 6]).astype(int)
    return df

def get_series_and_metrics_from_data():
    forecast = get_next_n_hours(24)
    if not forecast:
         return [], {
            "peakLoad": 0,
            "peakHour": "N/A",
            "avgLoad": 0,
            "changeVsYesterday": 0,
            "currentLoadSLDC": None 
        }

    df_pred = pd.DataFrame(forecast)
    if "predicted_load" not in df_pred.columns:
        return [], None
        
    peak_idx = df_pred["predicted_load"].idxmax()
    peak_row = df_pred.iloc[peak_idx]
    peak_val = peak_row["predicted_load"]
    peak_hour_int = peak_row["hour"]
    peak_hour_str = f"{peak_hour_int:02d}:00"
    
    avg_load = df_pred["predicted_load"].mean()
    
    change_pct = 0.0
    test_path = PROCESSED_DIR / "test_preprocessed.csv"
    if test_path.exists():
        try:
            df_hist = pd.read_csv(test_path, parse_dates=["datetime"])
            demand_col = "Power demand"
            if demand_col in df_hist.columns:
                 last_24h_actual = df_hist.tail(24 * 12)
                 prev_avg = last_24h_actual[demand_col].mean()
                 if prev_avg > 0:
                     change_pct = ((avg_load - prev_avg) / prev_avg) * 100
        except Exception:
            pass

    metrics = {
        "peakLoad": round(peak_val, 2),
        "peakHour": peak_hour_str,
        "avgLoad": round(avg_load, 2),
        "changeVsYesterday": round(change_pct, 2),
        "currentLoadSLDC": None
    }
    
    return forecast, metrics

def _fetch_tomorrow_forecast(cnt: int = 24) -> Optional[List[dict]]:
    if not TOMORROW_API_KEY:
        return None
    try:
        url = (
            f"{TOMORROW_FORECAST_URL}?location={LAT_DELHI},{LON_DELHI}&apikey={TOMORROW_API_KEY}"
        )
        req = urllib.request.Request(url)
        req.add_header('accept', 'application/json')
        req.add_header('accept-encoding', 'deflate, gzip, br')
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            ce = resp.info().get('Content-Encoding')
            raw_data = resp.read()
            if ce == 'gzip':
                import gzip
                data_str = gzip.decompress(raw_data).decode('utf-8')
            elif ce == 'br':
                 try:
                     import brotli
                     data_str = brotli.decompress(raw_data).decode('utf-8')
                 except ImportError:
                      data_str = raw_data.decode('utf-8', errors='ignore') 
            else:
                data_str = raw_data.decode('utf-8')

            data = json.loads(data_str)

        timeline = data.get("timelines", {})
        hourly = timeline.get("hourly", [])
        
        out = []
        for item in hourly[:cnt]:
            vals = item.get("values", {})
            temp = vals.get("temperature", 30)
            rhum = vals.get("humidity", 50)
            dwpt = vals.get("dewPoint", temp - (100 - rhum) / 5)
            wspd = vals.get("windSpeed", 3)
            wdir = vals.get("windDirection", 180)
            pres = vals.get("pressureSeaLevel", 1010)
            out.append({
                "temp": float(temp),
                "rhum": float(rhum),
                "dwpt": float(dwpt),
                "wspd": float(wspd),
                "wdir": float(wdir),
                "pres": float(pres),
            })
        return out if out else None
    except Exception as e:
        print(f"Error fetching Tomorrow.io data: {e}")
        return None

def _weather_by_hour_ist(hour: int) -> dict:
    angle = 2 * np.pi * (hour - 5.5) / 24
    temp = 28 + 7 * np.sin(angle)
    rhum = 58 - 18 * np.sin(angle)
    rhum = max(35, min(85, rhum))
    wspd = 2.0 + 1.5 * (0.5 + 0.5 * np.sin(2 * np.pi * (hour - 14) / 24))
    wspd = max(0.5, min(5.0, wspd))
    wdir = 180 + 40 * np.sin(2 * np.pi * hour / 24)
    pres = 1008 + 5 * np.sin(2 * np.pi * (hour - 10) / 24)
    dwpt = temp - (100 - rhum) / 5
    return {"temp": float(temp), "rhum": float(rhum), "dwpt": float(dwpt), "wspd": float(wspd), "wdir": float(wdir), "pres": float(pres)}

def _fetch_live_load_sldc() -> Optional[float]:
    url = "https://www.delhisldc.org/Loadcurve.aspx"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8')
        
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            
        import re
        match = re.search(r'Delhi.*?<td[^>]*>([\d\.]+)</td>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return float(match.group(1))

        match = re.search(r'Delhi\s*Load\s*[:\-\s]*([\d\.]+)', html, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"SLDC Scrape Error: {e}")
        return None

def _get_gemini_adjustment(hour: int, temp: float, rhum: float) -> float:
    if not gemini_model:
        return 1.0
    try:
        prompt = (
            f"It is {hour}:00 in Delhi. Temperature is {temp}C, Humidity is {rhum}%. "
            "Electricity demand prediction needs balancing. "
            "Is this a time of high or low residential/commercial activity compared to average? "
            "Return ONLY a single float factor between 0.9 (lower demand) and 1.1 (higher demand). "
            "e.g. 0.95 for late night, 1.05 for peak evening. Do not output text, just the number."
        )
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        import re
        match = re.search(r"([\d\.]+)", text)
        if match:
            factor = float(match.group(1))
            return max(0.85, min(1.15, factor))
    except Exception as e:
        print(f"Gemini AI Error: {e}")
    return 1.0

def get_next_n_hours(n: int, temp: float = None, rhum: float = None):
    if model is None or not feature_columns:
        return []
    now = pd.Timestamp.utcnow()
    now = now + pd.Timedelta(hours=5, minutes=30)
    
    forecast_data = None
    if temp is None and rhum is None:
        forecast_data = _fetch_tomorrow_forecast(cnt=n)
        
    result = []
    
    ai_factor = 1.0
    if n > 0:
        if temp is not None and rhum is not None:
             t_curr, r_curr = temp, rhum
        else:
             if forecast_data:
                 t_curr, r_curr = forecast_data[0]["temp"], forecast_data[0]["rhum"]
             else:
                 now_ist = now
                 t_curr = _weather_by_hour_ist(now_ist.hour)["temp"]
                 r_curr = _weather_by_hour_ist(now_ist.hour)["rhum"]
        
        ai_factor = _get_gemini_adjustment(now.hour, t_curr, r_curr)

    result = []
    for i in range(n):
        t = now + pd.Timedelta(hours=i)
        hour_ist = int(t.hour)
        
        if forecast_data and i < len(forecast_data):
            w = forecast_data[i]
            t_val, r_val = w["temp"], w["rhum"]
        elif temp is not None and rhum is not None:
             w = _weather_by_hour_ist(hour_ist)
             t_val, r_val = temp, rhum
        else:
             w = _weather_by_hour_ist(hour_ist)
             t_val, r_val = w["temp"], w["rhum"]

        feats = build_features_for_datetime(t, temp=t_val, rhum=r_val)
        feats["temp"] = t_val
        feats["rhum"] = r_val
        feats["dwpt"] = w.get("dwpt", feats["dwpt"])
        feats["wspd"] = w.get("wspd", feats["wspd"])
        feats["wdir"] = w.get("wdir", feats["wdir"])
        feats["pres"] = w.get("pres", feats["pres"])
        
        X = pd.DataFrame([feats])[feature_columns].fillna(0)
        base_pred = float(model.predict(X)[0])
        
        final_pred = base_pred * CALIBRATION_FACTOR * ai_factor
        
        ts_iso = t.strftime("%Y-%m-%dT%H:%M:%S+05:30") if hasattr(t, "strftime") else t.isoformat()
        result.append({
            "timestamp": ts_iso,
            "hour": hour_ist,
            "temp": round(t_val, 1),
            "dwpt": round(feats["dwpt"], 1),
            "rhum": round(r_val, 0),
            "wdir": round(feats["wdir"], 0),
            "wspd": round(feats["wspd"], 1),
            "pres": round(feats["pres"], 1),
            "actual_load": None,
            "predicted_load": round(final_pred, 2),
        })
    return result

def get_next_10_hours(temp: float = None, rhum: float = None):
    return get_next_n_hours(10, temp=temp, rhum=rhum)

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/api/model-info")
def model_info():
    if not model_metadata:
        return {"model_loaded": False}
    return {
        "model_loaded": True,
        "validation_r2": round(model_metadata.get("val_r2", 0), 4),
        "validation_mape_pct": round(model_metadata.get("val_mape", 0), 2),
        "confidence_base": model_metadata.get("confidence_base", 0.92),
        "feature_count": len(feature_columns),
    }

@app.post("/api/predict", response_model=PredictResponse)
def predict_demand(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run MODEL/train_model.py first.")
    features = build_features_from_scenario(req.date, req.temp, req.humidity, req.scenario)
    X = pd.DataFrame([features])[feature_columns]
    pred = float(model.predict(X.fillna(0))[0])
    pred = pred * CALIBRATION_FACTOR
    confidence = float(model_metadata.get("confidence_base", 0.92))
    if 20 <= req.temp <= 45 and 20 <= req.humidity <= 90:
        confidence = min(0.97, confidence + 0.02)
    confidence = round(max(0.88, min(0.97, confidence)), 2)
    return PredictResponse(predicted_load=round(pred, 2), confidence=confidence)

@app.get("/api/series")
def get_series(temp: Optional[float] = None, rhum: Optional[float] = None):
    series = get_next_n_hours(24, temp=temp, rhum=rhum)
    next_10_hours = get_next_n_hours(10, temp=temp, rhum=rhum)
    return {"series": series, "next_10_hours": next_10_hours}

@app.get("/api/metrics")
def get_metrics():
    _, metrics = get_series_and_metrics_from_data()
    if metrics is None:
        raise HTTPException(status_code=503, detail="No processed data. Run MODEL/del.py and train_model.py.")
    metrics = metrics or {}
    
    live_load = _fetch_live_load_sldc()
    if live_load is not None:
        metrics["currentLoadSLDC"] = round(live_load, 2)
    else:
        metrics["currentLoadSLDC"] = None
        
    return metrics

@app.on_event("startup")
def startup():
    if load_model():
        print("Model loaded from", PROCESSED_DIR)
    else:
        print("WARNING: Model not found. Run MODEL/train_model.py and ensure demand_model.joblib exists.")

