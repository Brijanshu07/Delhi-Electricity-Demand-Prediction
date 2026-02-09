# Delhi Electricity Demand Predictor

A modern, responsive web application for predicting electricity demand in Delhi using real-time weather data and machine learning predictions.

## Features

- **Real-time Weather Integration**: Fetches current weather data from OpenWeather API
- **Interactive Predictions**: 24-hour load forecasting with visual charts
- **Scenario Simulation**: Test different weather scenarios and their impact on electricity demand
- **Dark/Light Mode**: Toggle between themes with persistent preference
- **Data Export**: Export charts as PNG and data tables as CSV
- **Responsive Design**: Beautiful UI that works on all device sizes
- **Premium Styling**: Glassmorphism effects, smooth animations, and modern design

## Tech Stack

- **React** (with JSX and Hooks)
- **Vite** (build tool)
- **TailwindCSS** (styling)
- **Recharts** (data visualization)
- **Lucide React** (icons)
- **html2canvas** (chart export)

## Getting Started

### Prerequisites

- Node.js 16+ installed
- OpenWeather API key (optional, falls back to mock data)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```

4. Add your OpenWeather API key to `.env`:
   ```
   VITE_OPENWEATHER_API_KEY=your_api_key_here
   VITE_API_BASE_URL=http://localhost:8000
   ```

   **To get an API key:**
   - Go to [OpenWeather API](https://openweathermap.org/api)
   - Sign up for a free account
   - Generate an API key
   - Paste it in your `.env` file

### Development

Run the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## Project Structure

```
src/
├── api/
│   └── apiClient.js          # API integration layer (ready for backend)
├── components/
│   ├── Header.jsx             # App header with theme toggle
│   ├── WeatherCard.jsx        # Real-time weather display
│   ├── PredictionChart.jsx    # 24-hour forecast visualization
│   ├── MetricsCards.jsx       # Key metrics display
│   ├── ScenarioPanel.jsx      # Scenario simulation controls
│   └── DataTable.jsx          # Hourly data table
├── hooks/
│   └── useTheme.js            # Theme management hook
├── mocks/
│   └── mockData.js            # Mock data for development
├── utils/
│   └── exportUtils.js         # Export functionality (PNG/CSV)
├── App.jsx                    # Main application component
├── main.jsx                   # Application entry point
└── index.css                  # Global styles with Tailwind
```

## Connecting to Python Backend

The application is designed to easily connect to a Python backend. Currently, it uses mock data, but you can integrate your ML model by:

1. **Set up your Python backend** (FastAPI recommended):
   ```python
   # Example FastAPI endpoint
   @app.post("/predict")
   async def predict(data: PredictionInput):
       # Your ML model prediction logic
       return {"predicted_load": 4500.2, "confidence": 0.89}
   ```

2. **Update the API base URL** in `.env`:
   ```
   VITE_API_BASE_URL=http://your-backend-url:8000
   ```

3. **Modify `src/api/apiClient.js`**:
   - Replace mock implementations with actual fetch calls
   - The file has TODO comments showing where to add real API calls
   - Example:
     ```javascript
     // TODO: Replace with real API call
     // const response = await fetch(`${API_BASE_URL}/predict`, {
     //   method: 'POST',
     //   headers: { 'Content-Type': 'application/json' },
     //   body: JSON.stringify(payload)
     // });
     // return await response.json();
     ```

## Data Models

### Weather Object
```javascript
{
  temp: 33.4,      // Temperature in Celsius
  dwpt: 22.1,      // Dew point in Celsius
  rhum: 48,        // Relative humidity (%)
  wdir: 210,       // Wind direction (degrees)
  wspd: 3.5,       // Wind speed (m/s)
  pres: 1008,      // Atmospheric pressure (hPa)
  year: 2025,
  month: 11,
  day: 19,
  hour: 15,
  minute: 30,
  description: 'Clear sky',
  icon: '01d'
}
```

### Hourly Data Row
```javascript
{
  timestamp: "2025-11-19T15:00:00",
  temp: 33.4,
  dwpt: 22.1,
  rhum: 48,
  wdir: 210,
  wspd: 3.5,
  pres: 1008,
  actual_load: 4200.5,      // Actual electricity load (MW)
  predicted_load: 4350.2    // Predicted electricity load (MW)
}
```

## Features Breakdown

### Weather Card
- Displays real-time weather data for Delhi
- Fetches from OpenWeather API or uses mock data
- Shows temperature, humidity, wind, pressure, etc.
- Animated entrance effect
- Tooltips for field explanations

### Prediction Chart
- Overlaid line chart showing:
  - Last 24h actual load (blue line)
  - Next 24h predicted load (green dashed line)
- Smooth animations
- Custom tooltips
- Export to PNG functionality

### Scenario Panel
- Date picker for scenario testing
- Toggle between Normal/High Temperature scenarios
- Adjustable temperature and humidity sliders
- Real-time prediction simulation

### Metrics Cards
- Peak predicted load
- Peak hour
- % change vs yesterday
- Average load
- Animated on load

### Data Table
- Shows latest hourly data
- Scrollable and responsive
- Export to CSV functionality

## License

MIT

## Support

For issues or questions, please open an issue in the repository.
