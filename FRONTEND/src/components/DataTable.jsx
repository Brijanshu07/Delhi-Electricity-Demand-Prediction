import { Download } from 'lucide-react';
import { exportToCSV } from '../utils/exportUtils';

const DataTable = ({ data, loading }) => {
  const handleExport = () => {
    const exportData = data.map(row => ({
      timestamp: row.timestamp,
      temperature: row.temp,
      humidity: row.rhum,
      wind_speed: row.wspd,
      pressure: row.pres,

      predicted_load: row.predicted_load
    }));
    exportToCSV(exportData, 'delhi-electricity-data.csv');
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 animate-pulse">
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-12 bg-gray-200 dark:bg-gray-700 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  const displayData = Array.isArray(data) ? data.slice(0, 10) : [];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Hourly Forecast
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Next {displayData.length} hours from now — predicted load
          </p>
        </div>
        <button
          onClick={handleExport}
          className="flex items-center space-x-2 px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors duration-200"
        >
          <Download className="w-4 h-4" />
          <span className="text-sm font-medium">Export CSV</span>
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                Time
              </th>
              <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                Temp (°C)
              </th>
              <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                Humidity (%)
              </th>
              <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                Wind (m/s)
              </th>

              <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                Predicted (MW)
              </th>
            </tr>
          </thead>
          <tbody>
            {displayData.map((row, index) => {
              const time = new Date(row.timestamp);
              return (
                <tr
                  key={index}
                  className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <td className="py-3 px-4 text-gray-900 dark:text-white">
                    {time.toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" })}
                  </td>
                  <td className="py-3 px-4 text-right text-gray-700 dark:text-gray-300">
                    {row.temp.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-right text-gray-700 dark:text-gray-300">
                    {row.rhum.toFixed(0)}
                  </td>
                  <td className="py-3 px-4 text-right text-gray-700 dark:text-gray-300">
                    {row.wspd.toFixed(1)}
                  </td>

                  <td className="py-3 px-4 text-right text-green-600 dark:text-green-400 font-medium">
                    {row.predicted_load ? row.predicted_load.toFixed(1) : '-'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DataTable;
