# ⚡ Delhi Electricity Load Prediction System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB" alt="React"/>
  <img src="https://img.shields.io/badge/XGBoost-1b2024?style=flat" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Gemini_AI-Google-orange" alt="Gemini AI"/>
</div>

<br/>

A comprehensive, AI-driven machine learning system that predicts electricity load for Delhi with high accuracy. This project leverages an **XGBoost Regressor** enhanced with real-time AI adjustments via **Google Gemini** and live weather data from **Tomorrow.io**. 

Includes a fully interactive React dashboard to visualize 24-hour forecasts, key metrics, historical comparisons, and live data scraping from the Delhi SLDC website.

---

## ✨ Key Features

- **⏱️ 24-Hour Load Forecast**: Predicts electricity demand for the next 24 hours with hourly granularity.
- **🌤️ Real-Time Weather Integration**: Fetches live weather data from Tomorrow.io API for accurate, up-to-the-minute predictions.
- **🧠 AI-Enhanced Predictions**: Uses Google Gemini 1.5 Flash to apply contextual balancing factors based on time of day and weather intensity.
- **📊 Interactive Dashboard**: Modern React-based frontend featuring rich charts, tables, and metric visualizations.
- **🔄 Live Load Monitoring**: Automatically scrapes current load data straight from the Delhi SLDC website.
- **🎯 Scenario Analysis**: Supports testing different prediction scenarios (e.g., simulating high-temperature spikes).
- **✅ High Accuracy Validation**: Model boasts an R² > 0.9 on validation data and a MAPE < 5%.
- **🚀 Production Ready**: Pre-configured for seamless deployment to Render (Backend) and Netlify (Frontend).

## 🏆 Why This Model Excels

1. **Advanced Feature Engineering**: Incorporates cyclical time features (sin/cos for hour/month), cooling degree hours, heat discomfort proxies, and seasonal indicators to perfectly capture Delhi's unique demand patterns.
2. **Ensemble Learning**: Uses XGBoost (with Gradient Boosting fallback) for robust predictions, effectively handling non-linear relationships.
3. **AI Contextual Adjustment**: Gemini AI provides dynamic factors (0.9-1.1) to fine-tune predictions for subtle behavioral or weather shifts, improving reliability beyond purely numerical models.
4. **Calibration and Validation**: Static calibration factor (0.85) aligns outputs with recent trends. Validation metrics show very low error rates (RMSE ~50-100 MW, MAPE <5%).
5. **Confidence Scoring**: Delivers dynamic prediction confidence percentages (88-97%) depending on the stability of input conditions.

---

## 💻 Technology Stack

### Backend
- **Core language:** Python 3.9+
- **Framework:** FastAPI (high-performance async web API)
- **ML Libraries:** XGBoost (primary) / Scikit-Learn GradientBoostingRegressor, Pandas, NumPy, Joblib
- **AI Integration:** Google Generative AI SDK (Gemini 1.5 Flash)
- **External Data:** Tomorrow.io (weather APIs), Delhi SLDC (live load scraper)

### Frontend
- **Framework:** React.js (via Vite)
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **Icons:** Lucide React

### Infrastructure & Deployment
- **Backend Hosting:** Render
- **Frontend Hosting:** Netlify
- **Version Control:** Git

---

## 📂 Project Structure

```text
/
├── BACKEND/                 # Python FastAPI backend
│   ├── main.py              # Main API server
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables (API keys)
├── FRONTEND/                # React frontend
│   ├── src/
│   │   ├── components/      # React components (Dashboard, Charts, etc.)
│   │   ├── hooks/           # Custom React hooks
│   │   ├── mocks/           # Mock data for development
│   │   └── utils/           # Utility functions
│   ├── package.json         # Node.js dependencies
│   └── vite.config.js       # Vite configuration
├── MODEL/                   # ML model training and data pipeline
│   ├── train_model.py       # Model training script
│   ├── del.py               # Data preprocessing script
│   ├── processed/           # Saved trained models and metadata
│   └── data/                # Raw and processed datasets
├── render.yaml              # Render deployment configuration
├── netlify.toml             # Netlify deployment configuration
└── README.md                # Project documentation
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+ installed
- Node.js & npm installed
- API Keys:
  - **Google Gemini AI** (from Google AI Studio)
  - **Tomorrow.io** (Weather API)

### Local Development

#### 1. Backend Setup
```bash
cd BACKEND
```
Create a `.env` file and add your keys:
```env
TOMORROW_API_KEY=your_tomorrow_io_key
GEMINI_API_KEY=your_gemini_api_key
```
Install dependencies and run the server:
```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
> API will be available at `http://localhost:8000`.

#### 2. Frontend Setup
```bash
cd FRONTEND
npm install
npm run dev
```
> Access the dashboard at `http://localhost:5173`.

#### 3. Model Training (Optional)
If you wish to retrain the ML model on new data:
```bash
cd MODEL
python del.py           # Preprocess dataset
python train_model.py   # Train and save the new model
```

---

## ☁️ Production Deployment

### Backend (Render)
1. Sign up at [Render](https://render.com).
2. Connect your Git repository & create a new **Web Service**.
3. **Runtime:** Python 3
4. **Build Command:** `pip install -r BACKEND/requirements.txt`
5. **Start Command:** `cd BACKEND && uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Set `TOMORROW_API_KEY` and `GEMINI_API_KEY` as environment variables.
7. Deploy!

### Frontend (Netlify)
1. Sign up at [Netlify](https://netlify.com).
2. Connect your Git repository & create a new **Site**.
3. **Base directory:** `FRONTEND`
4. **Build command:** `npm run build`
5. **Publish directory:** `FRONTEND/dist`
6. Set the `VITE_API_BASE_URL` env variable to your mapped Render backend URL.
7. Deploy!

---

## 🔌 API Endpoints
- `GET /api/health` - Health check and model status
- `GET /api/model-info` - Model metadata (R², MAPE, confidence thresholds)
- `POST /api/predict` - Make a single prediction with a simulated scenario
- `GET /api/series` - Fetch 24-hour forecast time series
- `GET /api/metrics` - High-level metrics (Peak load, averages, WoW changes)

---

## ❓ Frequently Asked Questions (FAQs)

**Q: How accurate are the predictions?**  
**A:** The XGBoost model achieves an R² > 0.9 on validation sets. The Gemini AI integration further contextualizes predictions, maintaining a MAPE of < 5% for most scenarios.

**Q: What happens if the external APIs go down?**  
**A:** The system seamlessly falls back to simulated weather data, and Gemini scaling safely defaults to a `1.0` multiplier factor. The SLDC web scraper gracefully degrades to "N/A" if the official state site drops.

**Q: Can I run this system without API keys?**  
**A:** Yes, but feature accuracy will drop. It will utilize simulated weather logic and skip the AI adjustment phase entirely.

**Q: What is the generic calibration factor for?**  
**A:** The `0.85` calibration factor aligns base numerical predictions with sudden short-term observed trends. This can be dynamically adjusted based on recent performance.

---

## 🤝 Contributing
1. Fork the repository
2. Create your own feature branch
3. Test changes locally
4. Submit a descriptive Pull Request

## 📄 License
This project is open-source and available under the **MIT License**.
