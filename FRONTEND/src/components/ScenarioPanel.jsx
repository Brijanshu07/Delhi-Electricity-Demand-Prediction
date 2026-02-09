import { useState } from 'react';
import { Calendar, Thermometer, Droplets, Play } from 'lucide-react';
import { predict } from '../api/apiClient';

const ScenarioPanel = ({ onPredictionResult }) => {
  const [scenario, setScenario] = useState('normal');
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [temp, setTemp] = useState(33);
  const [humidity, setHumidity] = useState(50);
  const [loading, setLoading] = useState(false);

  const handleRunScenario = async () => {
    setLoading(true);
    try {
      const result = await predict({
        date,
        scenario,
        temp,
        humidity
      });
      onPredictionResult?.(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 animate-fade-in">
      <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
        Scenario Simulator
      </h2>

      <div className="space-y-4">
        <div>
          <label className="flex items-center space-x-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <Calendar className="w-4 h-4" />
            <span>Date</span>
          </label>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Scenario Type
          </label>
          <div className="flex space-x-2">
            <button
              onClick={() => setScenario('normal')}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-all ${
                scenario === 'normal'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              Normal
            </button>
            <button
              onClick={() => setScenario('high-temp')}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-all ${
                scenario === 'high-temp'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              High Temp
            </button>
          </div>
        </div>

        <div>
          <label className="flex items-center space-x-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <Thermometer className="w-4 h-4" />
            <span>Temperature: {temp}°C</span>
          </label>
          <input
            type="range"
            min="20"
            max="50"
            value={temp}
            onChange={(e) => setTemp(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>

        <div>
          <label className="flex items-center space-x-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <Droplets className="w-4 h-4" />
            <span>Humidity: {humidity}%</span>
          </label>
          <input
            type="range"
            min="20"
            max="90"
            value={humidity}
            onChange={(e) => setHumidity(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>

        <button
          onClick={handleRunScenario}
          disabled={loading}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-medium rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Running...</span>
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              <span>Run Scenario</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default ScenarioPanel;
