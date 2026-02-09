export const mockWeather = {
  temp: 33.4,
  dwpt: 22.1,
  rhum: 48,
  wdir: 210,
  wspd: 3.5,
  pres: 1008,
  year: 2025,
  month: 11,
  day: 19,
  hour: 15,
  minute: 30,
  description: 'Clear sky',
  icon: '01d'
};

export const generateMockSeries = () => {
  const data = [];
  const now = new Date();

  for (let i = -24; i < 24; i++) {
    const timestamp = new Date(now.getTime() + i * 60 * 60 * 1000);
    const baseLoad = 4000 + Math.sin(i / 3) * 800 + Math.random() * 200;

    data.push({
      timestamp: timestamp.toISOString(),
      hour: timestamp.getHours(),
      temp: 30 + Math.random() * 8,
      dwpt: 20 + Math.random() * 5,
      rhum: 40 + Math.random() * 20,
      wdir: 180 + Math.random() * 60,
      wspd: 2 + Math.random() * 4,
      pres: 1005 + Math.random() * 10,
      actual_load: i < 0 ? baseLoad : null,
      predicted_load: i >= 0 ? baseLoad + (Math.random() * 100 - 50) : null
    });
  }

  return data;
};

/** Next 10 hours from now (for table). */
export const generateMockNext10Hours = () => {
  const data = [];
  const now = new Date();
  for (let i = 0; i < 10; i++) {
    const timestamp = new Date(now.getTime() + i * 60 * 60 * 1000);
    const baseLoad = 4000 + Math.sin(i / 2) * 400 + Math.random() * 100;
    data.push({
      timestamp: timestamp.toISOString(),
      hour: timestamp.getHours(),
      temp: 30 + Math.random() * 6,
      dwpt: 20 + Math.random() * 4,
      rhum: 45 + Math.random() * 15,
      wdir: 180,
      wspd: 2.5,
      pres: 1008,
      actual_load: null,
      predicted_load: baseLoad
    });
  }
  return data;
};

export const mockMetrics = {
  peakLoad: 4856.2,
  peakHour: '14:00',
  changeVsYesterday: 5.3,
  avgLoad: 4234.5
};
