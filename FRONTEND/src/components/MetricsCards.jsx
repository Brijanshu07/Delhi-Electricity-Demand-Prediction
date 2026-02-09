import { TrendingUp, Clock, Activity, Zap } from 'lucide-react';

const MetricsCards = ({ metrics, loading }) => {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 animate-pulse">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-3"></div>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
          </div>
        ))}
      </div>
    );
  }

  const cards = [
    {
      label: 'Peak Load',
      value: `${metrics.peakLoad.toFixed(1)} MW`,
      icon: Zap,
      color: 'from-blue-500 to-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20'
    },
    {
      label: 'Peak Hour',
      value: metrics.peakHour,
      icon: Clock,
      color: 'from-green-500 to-green-600',
      bgColor: 'bg-green-50 dark:bg-green-900/20'
    },
    {
      label: 'Change vs Yesterday',
      value: `${metrics.changeVsYesterday > 0 ? '+' : ''}${metrics.changeVsYesterday.toFixed(1)}%`,
      icon: TrendingUp,
      color: metrics.changeVsYesterday >= 0 ? 'from-green-500 to-green-600' : 'from-red-500 to-red-600',
      bgColor: metrics.changeVsYesterday >= 0 ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'
    },
    {
      label: 'Average Load',
      value: `${metrics.avgLoad.toFixed(1)} MW`,
      icon: Activity,
      color: 'from-orange-500 to-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-900/20'
    },

  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, index) => {
        const Icon = card.icon;
        return (
          <div
            key={index}
            className={`${card.bgColor} rounded-xl shadow-lg p-6 hover:shadow-xl transition-all duration-300 animate-slide-up`}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                {card.label}
              </span>
              <div className={`p-2 bg-gradient-to-br ${card.color} rounded-lg`}>
                <Icon className="w-5 h-5 text-white" />
              </div>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {card.value}
            </p>
          </div>
        );
      })}
    </div>
  );
};

export default MetricsCards;
