# ==============================================
# Delhi Electricity Demand Data Preprocessing
# For XGBoost Forecasting
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# -------------------------------
# 1. LOAD DATA
# -------------------------------

# Path to your dataset
file_path = "data/delhi_demand.csv"   # <- change name if needed

# Load the data
df = pd.read_csv(file_path)

# Show first few rows
print("Initial Data:")
print(df.head())

# -------------------------------
# 2. CLEAN BASIC STRUCTURE
# -------------------------------

# Remove unnamed index column if exists
if 'Unnamed: 0' in df.columns or df.columns[0].lower() == 'unnamed: 0':
    df = df.drop(df.columns[0], axis=1)

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort by time (just in case)
df = df.sort_values('datetime')

# Set datetime as index
df = df.set_index('datetime')

# -------------------------------
# 3. CHECK AND HANDLE MISSING VALUES
# -------------------------------

# Show missing count
print("\nMissing values before filling:")
print(df.isna().sum())

# Interpolate numeric missing values (like moving_avg_3)
df = df.interpolate(method='linear')

# Forward fill any remaining NaN
df = df.fillna(method='ffill')

# -------------------------------
# 4. FIX OUTLIERS (optional)
# -------------------------------

# Clip 'Power demand' to remove extreme spikes
q1 = df['Power demand'].quantile(0.25)
q3 = df['Power demand'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df['Power demand'] = df['Power demand'].clip(lower, upper)

# -------------------------------
# 5. ADD TIME FEATURES
# -------------------------------

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding for hour (to capture daily pattern)
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

# -------------------------------
# 6. CREATE LAG AND ROLLING FEATURES
# -------------------------------

# Create lag features (previous readings)
for lag in [1, 2, 3, 6, 12, 24, 48, 288]:  # 288 = 1 day (24*12 five-minute intervals)
    df[f'lag_{lag}'] = df['Power demand'].shift(lag)

# Rolling averages (smoothed demand)
df['rolling_mean_6'] = df['Power demand'].rolling(window=6).mean()   # last 30 mins
df['rolling_mean_12'] = df['Power demand'].rolling(window=12).mean() # last 1 hour
df['rolling_mean_288'] = df['Power demand'].rolling(window=288).mean() # last day

# Drop rows with NaN caused by lagging
df = df.dropna()

# -------------------------------
# 7. DEFINE TARGET VARIABLE
# -------------------------------

# Predict 1 step (5 min) ahead
df['target'] = df['Power demand'].shift(-1)

# Drop last NaN row (from shift)
df = df.dropna()

# -------------------------------
# 8. SPLIT TRAIN AND TEST (time-based)
# -------------------------------

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

# -------------------------------
# 9. VISUALIZE (optional but useful)
# -------------------------------

plt.figure(figsize=(10,4))
plt.plot(df['Power demand'], label='Power demand')
plt.title("Delhi Electricity Demand Over Time")
plt.ylabel("Demand (kW)")
plt.legend()
plt.show()

sns.heatmap(df.isna(), cbar=False)
plt.title("Missing Values After Cleaning")
plt.show()

sns.boxplot(x=df['Power demand'])
plt.title("Power Demand Distribution After Outlier Fix")
plt.show()

# Optional: seasonal decomposition
try:
    result = seasonal_decompose(df['Power demand'], model='additive', period=288)
    result.plot()
    plt.show()
except Exception as e:
    print("Skipping seasonal decomposition (dataset too small for full day period).")

# -------------------------------
# 10. SAVE CLEANED DATA
# -------------------------------

os.makedirs("processed", exist_ok=True)
train_df.to_csv("processed/train_preprocessed.csv")
test_df.to_csv("processed/test_preprocessed.csv")

print("\n✅ Preprocessing complete! Files saved in 'processed/' folder.")
