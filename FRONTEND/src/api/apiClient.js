import {
  mockWeather,
  generateMockSeries,
  generateMockNext10Hours,
  mockMetrics,
} from "../mocks/mockData";

const OPENWEATHER_API_KEY = import.meta.env.VITE_OPENWEATHER_API_KEY;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

const api = (path, options = {}) => {
  if (!API_BASE_URL) return null;
  return fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
};

export const fetchWeather = async () => {
  if (!OPENWEATHER_API_KEY) {
    console.warn("OpenWeather API key not found, using mock data");
    return mockWeather;
  }

  try {
    const response = await fetch(
      `https://api.openweathermap.org/data/2.5/weather?q=Delhi,IN&appid=${OPENWEATHER_API_KEY}&units=metric`
    );

    if (!response.ok) {
      throw new Error("Weather API failed");
    }

    const data = await response.json();
    const now = new Date();

    return {
      temp: data.main.temp,
      dwpt: data.main.temp - (100 - data.main.humidity) / 5,
      rhum: data.main.humidity,
      wdir: data.wind.deg,
      wspd: data.wind.speed,
      pres: data.main.pressure,
      year: now.getFullYear(),
      month: now.getMonth() + 1,
      day: now.getDate(),
      hour: now.getHours(),
      minute: now.getMinutes(),
      description: data.weather[0].description,
      icon: data.weather[0].icon,
    };
  } catch (error) {
    console.error("Failed to fetch weather:", error);
    return mockWeather;
  }
};

/** Get time series for chart + next 10 hours for table. Returns { series, next_10_hours }. */
export const getSeries = async (temp, rhum) => {
  const fallback = () => ({
    series: generateMockSeries(),
    next_10_hours: generateMockNext10Hours(),
  });
  if (!API_BASE_URL) {
    return new Promise((resolve) => setTimeout(() => resolve(fallback()), 500));
  }
  try {
    const params = new URLSearchParams();
    if (temp != null) params.set("temp", temp);
    if (rhum != null) params.set("rhum", rhum);
    const qs = params.toString() ? `?${params.toString()}` : "";
    const res = await api(`/api/series${qs}`);
    if (!res.ok) throw new Error("Series failed");
    const data = await res.json();
    return {
      series: Array.isArray(data.series) ? data.series : data.series || [],
      next_10_hours: Array.isArray(data.next_10_hours) ? data.next_10_hours : data.next_10_hours || [],
    };
  } catch (e) {
    console.warn("Backend series failed, using mock:", e.message);
    return fallback();
  }
};

/** @deprecated Use getSeries */
export const getMockSeries = () => getSeries();

export const predict = async (payload) => {
  if (!API_BASE_URL) {
    await new Promise((r) => setTimeout(r, 800));
    return {
      predicted_load: 4200 + Math.random() * 500,
      confidence: 0.85 + Math.random() * 0.1,
    };
  }
  try {
    const res = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(res.statusText || "Prediction failed");
    return await res.json();
  } catch (e) {
    console.error("Prediction failed:", e);
    throw e;
  }
};

export const getMetrics = async () => {
  if (!API_BASE_URL) {
    await new Promise((r) => setTimeout(r, 300));
    return mockMetrics;
  }
  try {
    const res = await api("/api/metrics");
    if (!res.ok) throw new Error("Metrics failed");
    return await res.json();
  } catch (e) {
    console.warn("Backend metrics failed, using mock:", e.message);
    return mockMetrics;
  }
};
