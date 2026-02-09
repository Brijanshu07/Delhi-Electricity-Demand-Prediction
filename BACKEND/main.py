# import json
# import os
# import urllib.request
# from pathlib import Path
# from typing import Optional, List
# import numpy as np
# import pandas as pd
# import joblib
# import xgboost as xgb

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# # ---------------- CONFIG ----------------
# LAT_DELHI = 28.6128
# LON_DELHI = 77.2311
# CALIBRATION_FACTOR = 0.85

# TOMORROW_API_KEY = os.environ.get("TOMORROW_API_KEY")
# TOMORROW_FORECAST_URL = "https://api.tomorrow.io/v4/weather/forecast"

# BACKEND_DIR = Path(__file__).resolve().parent
# MODEL_DIR = BACKEND_DIR.parent / "MODEL" / "processed"

# # ---------------- APP ----------------
# app = FastAPI(title="Delhi Electricity Demand API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------- MODEL ----------------
# model = None
# booster = None
# feature_columns: List[str] = []
# model_metadata = {}

# def load_model():
#     global model, booster, feature_columns, model_metadata

#     model_path = MODEL_DIR / "demand_model.joblib"
#     if not model_path.exists():
#         return False

#     model = joblib.load(model_path)
#     booster = model.get_booster()

#     with open(MODEL_DIR / "feature_list.json") as f:
#         feature_columns[:] = json.load(f)

#     meta = MODEL_DIR / "model_metadata.json"
#     if meta.exists():
#         with open(meta) as f:
#             model_metadata.update(json.load(f))

#     return True

# # ---------------- SCHEMAS ----------------
# class PredictRequest(BaseModel):
#     date: str
#     scenario: str
#     temp: float
#     humidity: float

# class PredictResponse(BaseModel):
#     predicted_load: float
#     confidence: float

# # ---------------- WEATHER ----------------
# def fallback_weather(hour: int):
#     temp = 28 + 7 * np.sin(2 * np.pi * (hour - 6) / 24)
#     rhum = max(35, min(85, 60 - 15 * np.sin(2 * np.pi * hour / 24)))
#     return {
#         "temp": round(float(temp), 1),
#         "rhum": round(float(rhum), 0),
#         "dwpt": round(float(temp - (100 - rhum) / 5), 1),
#         "wspd": 3.0,
#         "wdir": 180.0,
#         "pres": 1010.0,
#     }

# def fetch_tomorrow_forecast(hours=24):
#     if not TOMORROW_API_KEY:
#         return None

#     try:
#         url = (
#             f"{TOMORROW_FORECAST_URL}"
#             f"?location={LAT_DELHI},{LON_DELHI}"
#             f"&apikey={TOMORROW_API_KEY}"
#             f"&timesteps=1h"
#         )

#         with urllib.request.urlopen(url, timeout=10) as r:
#             data = json.loads(r.read().decode())

#         hourly = data["timelines"]["hourly"][:hours]
#         out = []

#         for h in hourly:
#             v = h["values"]
#             out.append({
#                 "temp": v.get("temperature", 30),
#                 "rhum": v.get("humidity", 50),
#                 "dwpt": v.get("dewPoint", 20),
#                 "wspd": v.get("windSpeed", 3),
#                 "wdir": v.get("windDirection", 180),
#                 "pres": v.get("pressureSeaLevel", 1010),
#             })
#         return out

#     except Exception as e:
#         print("Tomorrow API failed:", e)
#         return None

# # ---------------- FEATURES ----------------
# def build_features(ts, w):
#     hour = ts.hour
#     month = ts.month
#     dow = ts.dayofweek

#     return {
#         "hour": hour,
#         "day_of_week": dow,
#         "is_weekend": int(dow >= 5),
#         "sin_hour": np.sin(2 * np.pi * hour / 24),
#         "cos_hour": np.cos(2 * np.pi * hour / 24),
#         "temp": w["temp"],
#         "rhum": w["rhum"],
#         "dwpt": w["dwpt"],
#         "wdir": w["wdir"],
#         "wspd": w["wspd"],
#         "pres": w["pres"],
#         "month": month,
#         "day": ts.day,
#         "sin_month": np.sin(2 * np.pi * month / 12),
#         "cos_month": np.cos(2 * np.pi * month / 12),
#         "cooling_degree": max(0, w["temp"] - 24),
#         "temp_x_rhum": w["temp"] * (w["rhum"] / 100),
#         "peak_hour": int(10 <= hour <= 18),
#         "summer_month": int(month in (4, 5, 6)),
#     }

# # ---------------- CORE ----------------
# def predict_series(n=24):
#     if not booster:
#         return []

#     now = pd.Timestamp.utcnow() + pd.Timedelta(hours=5, minutes=30)
#     weather = fetch_tomorrow_forecast(n)

#     series = []
#     for i in range(n):
#         ts = now + pd.Timedelta(hours=i)
#         w = weather[i] if weather and i < len(weather) else fallback_weather(ts.hour)

#         feats = build_features(ts, w)
#         X = pd.DataFrame([feats], columns=feature_columns).fillna(0)
#         dmat = xgb.DMatrix(X, feature_names=feature_columns)

#         pred = float(booster.predict(dmat)[0]) * CALIBRATION_FACTOR

#         series.append({
#             "timestamp": ts.isoformat(),
#             "hour": ts.hour,
#             "temp": w["temp"],
#             "rhum": w["rhum"],
#             "predicted_load": round(pred, 2),
#         })

#     return series

# # ---------------- ROUTES ----------------
# @app.get("/api/health")
# def health():
#     return {"status": "ok", "model_loaded": model is not None}

# @app.get("/api/series")
# def series():
#     data = predict_series(24)
#     return {"series": data, "next_10_hours": data[:10]}

# @app.get("/api/metrics")
# def metrics():
#     series = predict_series(24)
#     if not series:
#         return {
#             "peakLoad": 0,
#             "peakHour": "N/A",
#             "avgLoad": 0,
#             "changeVsYesterday": 0,
#             "currentLoadSLDC": None,
#         }

#     loads = [x["predicted_load"] for x in series]
#     peak = max(series, key=lambda x: x["predicted_load"])

#     return {
#         "peakLoad": round(max(loads), 2),
#         "peakHour": f"{peak['hour']:02d}:00",
#         "avgLoad": round(sum(loads) / len(loads), 2),
#         "changeVsYesterday": 0,
#         "currentLoadSLDC": None,
#     }

# @app.post("/api/predict", response_model=PredictResponse)
# def predict(req: PredictRequest):
#     ts = pd.to_datetime(req.date)
#     w = {
#         "temp": req.temp,
#         "rhum": req.humidity,
#         "dwpt": req.temp - (100 - req.humidity) / 5,
#         "wspd": 3,
#         "wdir": 180,
#         "pres": 1010,
#     }

#     feats = build_features(ts, w)
#     X = pd.DataFrame([feats], columns=feature_columns).fillna(0)
#     dmat = xgb.DMatrix(X, feature_names=feature_columns)

#     pred = float(booster.predict(dmat)[0]) * CALIBRATION_FACTOR

#     return PredictResponse(
#         predicted_load=round(pred, 2),
#         confidence=round(model_metadata.get("confidence_base", 0.92), 2),
#     )

# # ---------------- STARTUP ----------------
# @app.on_event("startup")
# def startup():
#     if load_model():
#         print("Model loaded")
#     else:
#         print("Model NOT found")


import json
import os
import urllib.request
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# CONFIG
# =========================
LAT_DELHI = 28.6128
LON_DELHI = 77.2311
CALIBRATION_FACTOR = 0.85

TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY", "")
TOMORROW_URL = "https://api.tomorrow.io/v4/weather/forecast"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "MODEL" / "processed"

# =========================
# LOAD MODEL
# =========================
model = None
booster = None
feature_columns: List[str] = []
model_metadata = {}

def load_model():
    global model, booster, feature_columns, model_metadata

    model_path = MODEL_DIR / "demand_model.joblib"
    if not model_path.exists():
        return False

    model = joblib.load(model_path)
    booster = model.get_booster()

    with open(MODEL_DIR / "feature_list.json") as f:
        feature_columns = json.load(f)

    meta = MODEL_DIR / "model_metadata.json"
    if meta.exists():
        with open(meta) as f:
            model_metadata = json.load(f)

    return True

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Delhi Electricity Demand API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SCHEMAS
# =========================
class PredictRequest(BaseModel):
    date: str
    scenario: str
    temp: float
    humidity: float

class PredictResponse(BaseModel):
    predicted_load: float
    confidence: float

# =========================
# WEATHER
# =========================
def fallback_weather(hour: int):
    angle = 2 * np.pi * (hour - 5.5) / 24
    temp = 28 + 7 * np.sin(angle)
    rhum = 55 - 15 * np.sin(angle)
    rhum = max(35, min(85, rhum))
    dwpt = temp - (100 - rhum) / 5

    return {
        "temp": float(temp),
        "rhum": float(rhum),
        "dwpt": float(dwpt),
        "wspd": 2.5,
        "wdir": 180.0,
        "pres": 1010.0,
    }

def fetch_tomorrow(n: int):
    if not TOMORROW_API_KEY:
        return None

    try:
        url = f"{TOMORROW_URL}?location={LAT_DELHI},{LON_DELHI}&apikey={TOMORROW_API_KEY}"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())

        hourly = data.get("timelines", {}).get("hourly", [])
        out = []

        for h in hourly[:n]:
            v = h["values"]
            out.append({
                "temp": float(v.get("temperature", 30)),
                "rhum": float(v.get("humidity", 50)),
                "dwpt": float(v.get("dewPoint", 20)),
                "wspd": float(v.get("windSpeed", 3)),
                "wdir": float(v.get("windDirection", 180)),
                "pres": float(v.get("pressureSeaLevel", 1010)),
            })

        return out
    except Exception:
        return None

# =========================
# FEATURES
# =========================
def build_features(ts: pd.Timestamp, w: dict):
    hour = ts.hour
    month = ts.month
    day = ts.day
    dow = ts.dayofweek

    return {
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "sin_hour": np.sin(2 * np.pi * hour / 24),
        "cos_hour": np.cos(2 * np.pi * hour / 24),
        "temp": w["temp"],
        "rhum": w["rhum"],
        "dwpt": w["dwpt"],
        "wdir": w["wdir"],
        "wspd": w["wspd"],
        "pres": w["pres"],
        "month": month,
        "day": day,
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
        "cooling_degree": max(0.0, w["temp"] - 24),
        "temp_x_rhum": w["temp"] * (w["rhum"] / 100),
        "peak_hour": int(10 <= hour <= 18),
        "summer_month": int(month in [4, 5, 6]),
    }

# =========================
# CORE PREDICTION
# =========================
def predict_hours(n: int):
    if model is None:
        return []

    now = pd.Timestamp.utcnow() + pd.Timedelta(hours=5, minutes=30)
    weather = fetch_tomorrow(n) or []

    result = []

    for i in range(n):
        ts = now + pd.Timedelta(hours=i)
        w = weather[i] if i < len(weather) else fallback_weather(ts.hour)

        feats = build_features(ts, w)
        X = pd.DataFrame([feats], columns=feature_columns).fillna(0)

        dmat = xgb.DMatrix(X, feature_names=booster.feature_names)
        pred = float(booster.predict(dmat)[0]) * CALIBRATION_FACTOR

        result.append({
            "timestamp": ts.isoformat(),
            "hour": ts.hour,
            "temp": round(w["temp"], 1),
            "dwpt": round(w["dwpt"], 1),
            "rhum": round(w["rhum"], 0),
            "wdir": round(w["wdir"], 0),
            "wspd": round(w["wspd"], 1),
            "pres": round(w["pres"], 1),
            "actual_load": 0.0,
            "predicted_load": round(pred, 2),
        })

    return result

# =========================
# ROUTES
# =========================
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/api/series")
def series():
    data = predict_hours(24)
    return {
        "series": data,
        "next_10_hours": data[:10]
    }

@app.get("/api/metrics")
def metrics():
    series = predict_hours(24)

    if not series:
        return {
            "peakLoad": 0.0,
            "peakHour": "00:00",
            "avgLoad": 0.0,
            "changeVsYesterday": 0.0,
            "currentLoadSLDC": 0.0,
        }

    loads = [x["predicted_load"] for x in series]
    peak_i = int(np.argmax(loads))

    return {
        "peakLoad": round(loads[peak_i], 2),
        "peakHour": f"{series[peak_i]['hour']:02d}:00",
        "avgLoad": round(float(np.mean(loads)), 2),
        "changeVsYesterday": 0.0,
        "currentLoadSLDC": 0.0,
    }

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ts = pd.to_datetime(req.date)
    w = {
        "temp": float(req.temp),
        "rhum": float(req.humidity),
        "dwpt": float(req.temp - (100 - req.humidity) / 5),
        "wspd": 3.0,
        "wdir": 180.0,
        "pres": 1010.0,
    }

    feats = build_features(ts, w)
    X = pd.DataFrame([feats], columns=feature_columns).fillna(0)

    dmat = xgb.DMatrix(X, feature_names=booster.feature_names)
    pred = float(booster.predict(dmat)[0]) * CALIBRATION_FACTOR

    return PredictResponse(
        predicted_load=round(pred, 2),
        confidence=round(float(model_metadata.get("confidence_base", 0.92)), 2)
    )

# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup():
    if not load_model():
        print("❌ Model not loaded")
    else:
        print("✅ Model loaded")
