# Delhi Electricity Load Prediction System

## Overview
This project predicts the electricity load for Delhi using a Machine Learning model (Random Forest Regressor) enhanced with real-time AI adjustments (Google Gemini) and live weather data (Tomorrow.io). The system provides an interactive dashboard to visualize the 24-hour forecast, key metrics, and historical comparisons.

## Project Flow
1.  **Data Collection**: 
    - Historical load data and weather parameters are used to train the model.
    - Real-time weather data is fetched from the **Tomorrow.io API**.
    - Live electricity load is scraped from the **Delhi SLDC website** (State Load Despatch Center).
    
2.  **Feature Engineering**: 
    - Input features include Date, Time, Temperature, Humidity, Wind Speed, and cyclical time features (sin/cos of hour/month).
    
3.  **Prediction Model**: 
    - A **Random Forest Regressor** predicts the base load based on the weather and time features.
    
4.  **AI Enhancement**: 
    - **Google Gemini AI** analyzes the context (e.g., specific time of day, weather intensity) to generate a dynamic "balancing factor" (e.g., 0.95 or 1.05) to fine-tune the prediction.
    - A static **Calibration Factor** (0.85) is applied to align model outputs with recent observed trends.

5.  **Visualization**: 
    - A React-based frontend displays the **Predicted Load Curve**, detailed hourly data table, and key metrics (Peak Load, Average Load, Change vs Yesterday).

## Deployment Steps

### Prerequisites
- Python 3.9+ installed
- Node.js & npm installed
- API Keys for:
    - Google Gemini AI
    - Tomorrow.io (Weather)

### 1. Backend Setup
Navigate to the `BACKEND` directory:
```bash
cd BACKEND
```

Create a `.env` file (if not exists) and add your keys:
```env
TOMORROW_API_KEY=your_tomorrow_io_key
GEMINI_API_KEY=your_gemini_api_key
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the server:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

### 2. Frontend Setup
Navigate to the `FRONTEND` directory:
```bash
cd ../FRONTEND
```

Install dependencies:
```bash
npm install
```

Run the development server:
```bash
npm run dev
```
Access the application at `http://localhost:5173` (or the port shown in the terminal).

## Technology Stack

### Backend
- **Language**: Python
- **Framework**: FastAPI (High-performance web framework)
- **ML Libraries**: Scikit-Learn (Random Forest), Pandas, NumPy, Joblib
- **AI Integration**: Google Generative AI SDK (Gemini)
- **External APIs**: Tomorrow.io (Weather), Delhi SLDC (Scraping)

### Frontend
- **Framework**: React.js (Vite)
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React

## Frequently Asked Questions (FAQs)

### Q: How accurate is the prediction?
A: The base Random Forest model has high accuracy (R² > 0.9). The Gemini AI integration further improves reliability by adjusting for subtle contextual factors that the numerical model might miss (e.g., sudden weather shifts or behavioral patterns).

### Q: What happens if the SLDC website is down?
A: The system implements a graceful fallback. The "Current Load" metric usually scraped from SLDC will show as "N/A" or not appear, but the prediction model will continue to function using Tomorrow.io weather data.

### Q: Can I run this without the API keys?
A: 
- Without `TOMORROW_API_KEY`: The system will use fallback/simulated weather data, reducing accuracy.
- Without `GEMINI_API_KEY`: The AI balancing factor defaults to 1.0 (no adjustment), so predictions will rely solely on the Random Forest model.

### Q: Where is the model file located?
A: The trained model (`demand_model.joblib`) is stored in the `MODEL/processed/` directory along with metadata `feature_list.json`.

### Q: How do I retrain the model?
A: Scripts in the `MODEL` folder (e.g., `train_model.py`) can be executed to retrain the model on new data. Ensure your historical dataset is up to date in the inputs folder.
