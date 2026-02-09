import { Cloud, Droplets, Wind, Gauge, Navigation } from 'lucide-react';
import { useState, useEffect } from 'react';

const WeatherCard = ({ weather, loading }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 animate-pulse">
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
        <div className="space-y-3">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
        </div>
      </div>
    );
  }

  if (!weather) return null;

  const weatherFields = [
    { label: 'Temperature', value: `${weather.temp}°C`, icon: Cloud, tooltip: 'Current temperature' },
    { label: 'Dew Point', value: `${weather.dwpt}°C`, icon: Droplets, tooltip: 'Dew point temperature' },
    { label: 'Humidity', value: `${weather.rhum}%`, icon: Droplets, tooltip: 'Relative humidity' },
    { label: 'Wind Speed', value: `${weather.wspd} m/s`, icon: Wind, tooltip: 'Wind speed' },
    { label: 'Wind Dir', value: `${weather.wdir}°`, icon: Navigation, tooltip: 'Wind direction' },
    { label: 'Pressure', value: `${weather.pres} hPa`, icon: Gauge, tooltip: 'Atmospheric pressure' }
  ];

  return (
    <div
      className={`bg-gradient-to-br from-blue-50 to-blue-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-lg p-6 transition-all duration-500 ${
        mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
      }`}
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
            Current Weather
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
            {weather.description || 'Delhi, India'}
          </p>
        </div>
        {weather.icon && (
          <img
            src={`https://openweathermap.org/img/wn/${weather.icon}@2x.png`}
            alt={weather.description}
            className="w-16 h-16"
          />
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {weatherFields.map((field, index) => {
          const Icon = field.icon;
          return (
            <div
              key={index}
              className="group relative bg-white dark:bg-gray-800 rounded-lg p-3 hover:shadow-md transition-all duration-200"
              title={field.tooltip}
            >
              <div className="flex items-center space-x-2 mb-1">
                <Icon className="w-4 h-4 text-blue-500" />
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {field.label}
                </span>
              </div>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {field.value}
              </p>
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                {field.tooltip}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-blue-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Last updated: {weather.year}-{String(weather.month).padStart(2, '0')}-{String(weather.day).padStart(2, '0')} {String(weather.hour).padStart(2, '0')}:{String(weather.minute).padStart(2, '0')}
        </p>
      </div>
    </div>
  );
};

export default WeatherCard;
