import { useState, useEffect } from 'react';
import Header from './components/Header';
import WeatherCard from './components/WeatherCard';
import PredictionChart from './components/PredictionChart';
import MetricsCards from './components/MetricsCards';
import ScenarioPanel from './components/ScenarioPanel';
import DataTable from './components/DataTable';
import { useTheme } from './hooks/useTheme';
import { fetchWeather, getSeries, getMetrics } from './api/apiClient';

function App() {
  const { theme, toggleTheme } = useTheme();
  const [weather, setWeather] = useState(null);
  const [seriesData, setSeriesData] = useState({ series: [], next_10_hours: [] });
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [weatherData, seriesPayload, metricsData] = await Promise.all([
        fetchWeather(),
        getSeries(),
        getMetrics()
      ]);

      setWeather(weatherData);
      setSeriesData(Array.isArray(seriesPayload) ? { series: seriesPayload, next_10_hours: [] } : { series: seriesPayload?.series ?? [], next_10_hours: seriesPayload?.next_10_hours ?? [] });
      setMetrics(metricsData);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-2">
            <WeatherCard weather={weather} loading={loading} />
          </div>
          <div>
            <ScenarioPanel onPredictionResult={setPredictionResult} />
          </div>
        </div>

        <div className="mb-6">
          <MetricsCards metrics={metrics} loading={loading} />
        </div>

        <div className="mb-6">
          <PredictionChart data={seriesData.series} loading={loading} />
        </div>

        {predictionResult && (
          <div className="mb-6 bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border-l-4 border-green-500 rounded-lg p-4 animate-slide-up">
            <h3 className="font-bold text-green-900 dark:text-green-100 mb-2">
              Scenario Prediction Result
            </h3>
            <p className="text-green-800 dark:text-green-200">
              Predicted Load: <span className="font-semibold">{predictionResult.predicted_load.toFixed(2)} MW</span>
            </p>
            <p className="text-sm text-green-700 dark:text-green-300">
              Confidence: {(predictionResult.confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}

        <div>
          <DataTable data={seriesData.next_10_hours} loading={loading} />
        </div>
      </main>
    </div>
  );
}

export default App;
