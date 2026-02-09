import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Download } from "lucide-react";
import { exportToPNG } from "../utils/exportUtils";

const formatTime = (timestamp) => {
  const d = new Date(timestamp);
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
};

const PredictionChart = ({ data, loading }) => {
  const chartData = (data || []).map((d) => ({
    time: formatTime(d.timestamp),
    hour: new Date(d.timestamp).getHours(),

    predicted: d.predicted_load,
  }));

  const handleExport = () => {
    exportToPNG("prediction-chart", "delhi-electricity-prediction.png");
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 animate-pulse">
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-80 bg-gray-200 dark:bg-gray-700 rounded"></div>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3">
        <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          Time: {payload[0].payload.time}
        </p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {entry.value ? entry.value.toFixed(2) : "N/A"} MW
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Load Forecast
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Next 24 hours from now — predicted load (IST)
          </p>
        </div>
        <button
          onClick={handleExport}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors duration-200"
        >
          <Download className="w-4 h-4" />
          <span className="text-sm font-medium">Export PNG</span>
        </button>
      </div>

      <div id="prediction-chart">
        <ResponsiveContainer width="100%" height={350}>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#e5e7eb"
              className="dark:stroke-gray-700"
            />
            <XAxis
              dataKey="time"
              stroke="#6b7280"
              className="dark:stroke-gray-400"
              tick={{ fill: "#6b7280", fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#6b7280"
              className="dark:stroke-gray-400"
              tick={{ fill: "#6b7280" }}
              label={{
                value: "Load (MW)",
                angle: -90,
                position: "insideLeft",
                fill: "#6b7280",
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ paddingTop: "20px" }} />

            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              name="Predicted Load"
              strokeDasharray="5 5"
              animationDuration={1000}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PredictionChart;
