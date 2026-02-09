# Delhi Electricity Load Prediction System

## Overview

This project predicts the electricity load for Delhi using a Machine Learning model (XGBoost Regressor) enhanced with real-time AI adjustments (Google Gemini) and live weather data (Tomorrow.io). The system provides an interactive dashboard to visualize the 24-hour forecast, key metrics, and historical comparisons. It integrates live data scraping from the Delhi SLDC website for current load monitoring.

## Key Features

- **24-Hour Load Forecast**: Predicts electricity demand for the next 24 hours with hourly granularity.
- **Real-Time Weather Integration**: Fetches live weather data from Tomorrow.io API for accurate predictions.
- **AI-Enhanced Predictions**: Uses Google Gemini AI to apply contextual balancing factors based on time of day and weather intensity.
- **Interactive Dashboard**: React-based frontend with charts, tables, and metrics visualization.
- **Scenario Analysis**: Supports different prediction scenarios (e.g., high-temperature adjustments).
- **Live Load Monitoring**: Scrapes current load data from Delhi SLDC website.
- **Model Validation**: High accuracy with R² > 0.9 on validation data, MAPE < 5%.
- **Deployment Ready**: Easy deployment to Render (backend) and Netlify (frontend).

## Why This Model is Better

- **Advanced Feature Engineering**: Incorporates cyclical time features (sin/cos for hour/month), cooling degree hours, heat discomfort proxies, and seasonal indicators to capture Delhi's unique demand patterns.
- **Ensemble Learning**: Uses XGBoost (or Gradient Boosting fallback) for robust predictions, handling non-linear relationships better than simple models.
- **AI Contextual Adjustment**: Gemini AI provides dynamic factors (0.9-1.1) to fine-tune predictions for subtle factors like behavioral patterns or sudden weather shifts, improving reliability beyond numerical models.
- **Calibration and Validation**: Static calibration factor (0.85) aligns outputs with recent trends. Validation metrics show low error rates (RMSE ~50-100 MW, MAPE <5%), outperforming basic time-series models.
- **Real-Time Data**: Integrates live weather and load data, reducing forecast errors compared to static models.
- **Confidence Scoring**: Provides prediction confidence (88-97%) based on input conditions, helping users assess reliability.

## Technology Stack

### Backend

- **Language**: Python 3.9+
- **Framework**: FastAPI (high-performance async web API)
- **ML Libraries**: XGBoost (primary) or Scikit-Learn GradientBoostingRegressor, Pandas, NumPy, Joblib
- **AI Integration**: Google Generative AI SDK (Gemini 1.5 Flash)
- **External APIs**: Tomorrow.io (weather), Delhi SLDC (live load scraping)
- **Other**: python-dotenv, urllib, ssl for secure requests

### Frontend

- **Framework**: React.js (with Vite for fast development)
- **Styling**: Tailwind CSS (utility-first CSS framework)
- **Charts**: Recharts (React charting library)
- **Icons**: Lucide React
- **Build Tool**: Vite (modern bundler)

### Deployment

- **Backend**: Render (cloud hosting for Python apps)
- **Frontend**: Netlify (static site hosting with CI/CD)
- **Version Control**: Git (GitHub/GitLab integration)

## Project Structure

```
/
├── BACKEND/                 # Python FastAPI backend
│   ├── main.py             # Main API server
│   ├── requirements.txt    # Python dependencies
│   └── .env                # Environment variables (API keys)
├── FRONTEND/               # React frontend
│   ├── src/
│   │   ├── components/     # React components (Dashboard, Charts, etc.)
│   │   ├── hooks/          # Custom React hooks
│   │   ├── mocks/          # Mock data for development
│   │   └── utils/          # Utility functions
│   ├── package.json        # Node.js dependencies
│   └── vite.config.js      # Vite configuration
├── MODEL/                  # ML model training and data
│   ├── train_model.py      # Model training script
│   ├── del.py              # Data preprocessing
│   ├── processed/          # Trained model and metadata
│   └── data/               # Raw and processed datasets
├── render.yaml             # Render deployment config
├── netlify.toml            # Netlify deployment config
└── README.md               # This file
```

## Setup Instructions

### Prerequisites

- Python 3.9+ installed
- Node.js & npm installed
- API Keys:
  - Google Gemini AI (from Google AI Studio)
  - Tomorrow.io (weather API)

### Local Development Setup

#### 1. Backend Setup

Navigate to the `BACKEND` directory:

```bash
cd BACKEND
```

Create a `.env` file with your API keys:

```env
TOMORROW_API_KEY=your_tomorrow_io_key
GEMINI_API_KEY=your_gemini_api_key
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the backend server:

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`.

#### 2. Frontend Setup

Navigate to the `FRONTEND` directory:

```bash
cd ../FRONTEND
```

Install Node.js dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Access the app at `http://localhost:5173` (default Vite port).

#### 3. Model Training (Optional)

If you need to retrain the model:

```bash
cd MODEL
python del.py  # Preprocess data
python train_model.py  # Train and save model
```

### Production Deployment

#### Backend (Render)

1. Sign up at [render.com](https://render.com).
2. Connect your Git repository.
3. Create a new Web Service:
   - Runtime: Python 3
   - Build Command: `pip install -r BACKEND/requirements.txt`
   - Start Command: `cd BACKEND && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set environment variables: `TOMORROW_API_KEY`, `GEMINI_API_KEY`.
5. Deploy and note the URL (e.g., `https://your-backend.onrender.com`).

#### Frontend (Netlify)

1. Sign up at [netlify.com](https://netlify.com).
2. Connect your Git repository.
3. Create a new site:
   - Base directory: `FRONTEND`
   - Build command: `npm run build`
   - Publish directory: `FRONTEND/dist`
4. Set environment variable: `VITE_API_BASE_URL=https://your-backend.onrender.com`
5. Deploy and access your site.

## API Endpoints

- `GET /api/health`: Health check and model status
- `GET /api/model-info`: Model metadata (R², MAPE, confidence)
- `POST /api/predict`: Single prediction with scenario
- `GET /api/series`: 24-hour forecast series
- `GET /api/metrics`: Key metrics (peak load, average, change vs yesterday)

## Model Details

### Training Data

- Historical electricity demand data from Delhi SLDC
- Weather parameters (temperature, humidity, wind, pressure)
- Time features (hour, day, month, cyclical encodings)

### Features Used

- Base: hour, day_of_week, is_weekend, sin_hour, cos_hour, temp, rhum, dwpt, wdir, wspd, pres
- Derived: month, day, sin_month, cos_month, cooling_degree, temp_x_rhum, peak_hour, summer_month

### Performance Metrics

- Validation R²: ~0.90-0.95
- Validation MAPE: <5%
- RMSE: ~50-100 MW
- Confidence: 88-97% (based on input conditions)

### Prediction Quality Advantages

- **Accuracy**: Ensemble model with rich features outperforms simple linear models.
- **Adaptability**: AI adjustments handle anomalies better than static models.
- **Real-Time**: Live data integration reduces lag and improves freshness.
- **Validation**: Time-based splits prevent data leakage; metrics ensure reliability.

## Frequently Asked Questions (FAQs)

### Q: How accurate is the prediction?

A: The XGBoost model achieves R² > 0.9 on validation. Gemini AI further refines predictions, resulting in MAPE < 5% for most scenarios.

### Q: What if external APIs are down?

A: Fallback to simulated weather data; Gemini defaults to 1.0 factor. SLDC scraping may show "N/A" for current load.

### Q: Can I run without API keys?

A: Yes, but accuracy drops. Without Tomorrow.io, uses simulated weather. Without Gemini, no AI adjustment.

### Q: How to update the model?

A: Run `MODEL/del.py` for preprocessing, then `MODEL/train_model.py` for training. Ensure new data in `MODEL/data/`.

### Q: What's the calibration factor?

A: 0.85 to align predictions with observed trends, adjustable based on recent performance.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and test locally.
4. Submit a pull request.

## License

This project is open-source under the MIT License.

## Contact

For questions or issues, open a GitHub issue or contact the maintainers.
