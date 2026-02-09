# preprocess_delhi_power.py
"""
Preprocessing pipeline for:
'Delhi 5-Minute Electricity Demand for Forecasting' (Kaggle)
Produces:
 - processed CSVs (train/val/test)
 - diagnostic plots (PNG)
 - saved scalers and metadata
"""
import os
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import holidays

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("data")
OUT_DIR = Path("processed")
PLOTS_DIR = Path("plots")
for d in (DATA_DIR, OUT_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Kaggle dataset identifiers (adjust if different)
KAGGLE_DATASET = "yug201/delhi-5-minute-electricity-demand-for-forecasting"
CSV_NAME_GUESS = "powerdemand_5min_2021_to_2024_with weather.csv"  # common notebook name

# -----------------------
# helper: download from Kaggle (optional)
# -----------------------
def download_kaggle_dataset(dataset_id: str, filename_hint: str):
    """
    Attempts to download the first CSV matching filename_hint using kaggle CLI.
    Requires kaggle.json in ~/.kaggle/.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("kaggle package not installed or not configured. Skipping download.")
        return None

    api = KaggleApi()
    api.authenticate()
    files = api.dataset_list_files(dataset_id).files
    file_candidates = [f.name for f in files if filename_hint.split()[0].lower() in f.name.lower()]
    if not file_candidates:
        file_candidates = [f.name for f in files if f.name.endswith('.csv')]
    if not file_candidates:
        print("No csv file found in dataset via Kaggle API.")
        return None

    fname = file_candidates[0]
    print("Downloading:", fname)
    out_path = DATA_DIR / fname
    if out_path.exists():
        print("File already exists:", out_path)
        return out_path
    api.dataset_download_file(dataset_id, fname, path=str(DATA_DIR), unzip=True)
    return out_path

# Try to find or download CSV
csv_path = None
local_guesses = list(DATA_DIR.glob("*.csv"))
if local_guesses:
    csv_path = local_guesses[0]
else:
    possible = download_kaggle_dataset(KAGGLE_DATASET, CSV_NAME_GUESS)
    if possible:
        csv_path = possible

if csv_path is None:
    print("No CSV found. Please download the dataset from Kaggle and place the CSV in the 'data/' folder.")
    sys.exit(1)

print("Loading CSV:", csv_path)
df = pd.read_csv(csv_path, parse_dates=['datetime'], dayfirst=False)

# -----------------------
# Basic checks
# -----------------------
print(df.shape)
print(df.columns.tolist())
print(df.head())

# Ensure target column name known (accept common variations)
target_col = None
for c in df.columns:
    if c.lower().strip() in ("power demand", "power_demand", "powerdemand", "demand", "power"):
        target_col = c
        break
if target_col is None:
    raise ValueError("Could not find 'Power demand' column. Columns: " + ", ".join(df.columns))

# -----------------------
# Indexing and frequency
# -----------------------
df = df.sort_values('datetime').reset_index(drop=True)
df = df.set_index('datetime')
# Reindex to strict 5-minute frequency to expose missing timestamps
full_index = pd.date_range(df.index.min(), df.index.max(), freq='5T')
df = df.reindex(full_index)
df.index.name = 'datetime'

# Keep original target (for plotting)
df[f'{target_col}_orig'] = df[target_col]

# Missing values summary
missing_summary = df.isna().sum()
print("Missing counts per column:\n", missing_summary)

# -----------------------
# Imputation strategy
# -----------------------
# 1) Short gaps: time interpolation
df[target_col] = df[target_col].interpolate(method='time', limit=12)  # up to 1 hour via interpolation

# 2) For longer gaps, fill with median for that time-of-day (hour+minute)
df['hour'] = df.index.hour
df['minute'] = df.index.minute
median_by_time = df.groupby(['hour', 'minute'])[target_col].transform('median')
df[target_col] = df[target_col].fillna(median_by_time)

# 3) If still NA (edge cases), forward fill then backward fill
df[target_col] = df[target_col].fillna(method='ffill').fillna(method='bfill')

# Add imputation flag
df['power_imputed_flag'] = df[f'{target_col}_orig'].isna().astype(int)

# For weather features: use time interpolation
weather_cols = [c for c in df.columns if c in ['temp','dwpt','rhum','wdir','wspd','pres']]
for c in weather_cols:
    df[c] = df[c].interpolate(method='time', limit=48).fillna(method='ffill').fillna(method='bfill')

# -----------------------
# Outlier detection (robust)
# -----------------------
# We'll compute daily MAD-based zscore on residual after subtracting rolling median (24h window)
df['rolling_med_288'] = df[target_col].rolling(window=288, min_periods=24, center=True).median()
resid = df[target_col] - df['rolling_med_288']
mad = resid.abs().rolling(window=288, min_periods=24, center=True).median()
# robust z
df['robust_z'] = 0.6745 * resid / (mad + 1e-9)
# flag outliers
df['is_outlier'] = (df['robust_z'].abs() > 6).astype(int)  # threshold 6 is conservative

# Replace outliers by rolling median
df.loc[df['is_outlier']==1, target_col] = df.loc[df['is_outlier']==1, 'rolling_med_288']

# -----------------------
# Feature engineering
# -----------------------
# Basic time features (already have hour/min)
df['dayofweek'] = df.index.dayofweek
df['day'] = df.index.day
df['month'] = df.index.month
df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

# cyclical encodings: daily & weekly
# daily period in minutes = 24*60 = 1440. For features we can use index minutes since midnight
minutes_in_day = (df.index.hour * 60 + df.index.minute).astype(int)
df['sin_day'] = np.sin(2*np.pi*minutes_in_day/1440)
df['cos_day'] = np.cos(2*np.pi*minutes_in_day/1440)
# weekly period in minutes = 7*1440
minutes_in_week = ((df.index.dayofweek * 1440) + minutes_in_day).astype(int)
df['sin_week'] = np.sin(2*np.pi*minutes_in_week/(7*1440))
df['cos_week'] = np.cos(2*np.pi*minutes_in_week/(7*1440))

# Lags - careful: shift so no leakage
lag_list = [1, 12, 24, 48, 72, 288, 2016]  # 5-min steps: 1=5min, 12=1hr, 288=1day, 2016=1week
for lag in lag_list:
    df[f'lag_{lag}'] = df[target_col].shift(lag)

# Rolling stats (based on past values only)
df['roll_mean_3'] = df[target_col].shift(1).rolling(window=3, min_periods=1).mean()
df['roll_mean_12'] = df[target_col].shift(1).rolling(window=12, min_periods=1).mean()
df['roll_std_12'] = df[target_col].shift(1).rolling(window=12, min_periods=1).std().fillna(0)
df['roll_mean_288'] = df[target_col].shift(1).rolling(window=288, min_periods=1).mean()

# Holiday feature (India)
ind_holidays = holidays.India()
df['is_holiday'] = df.index.date.astype('datetime64').astype('datetime64[ns]').map(lambda d: 1 if d in ind_holidays else 0)  # crude per-day holiday flag
df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

# Interaction examples
if 'temp' in df.columns:
    df['temp_x_wspd'] = df['temp'] * df['wspd']

# Drop rows with NA after feature creation (first few rows where lags missing)
df = df.dropna(subset=[f'lag_{l}' for l in lag_list])

# -----------------------
# Train/validation/test split (time-based)
# -----------------------
df = df.sort_index()
n = len(df)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

print("Shapes:", train_df.shape, val_df.shape, test_df.shape)

# -----------------------
# Scaling numeric features (RobustScaler)
# -----------------------
feature_cols = [
    # time / cyclical
    'sin_day','cos_day','sin_week','cos_week','dayofweek','is_weekend','is_holiday',
    # weather
] + [c for c in ['temp','dwpt','rhum','wdir','wspd','pres'] if c in df.columns] +  [f'lag_{l}' for l in lag_list] + ['roll_mean_3','roll_mean_12','roll_std_12','roll_mean_288','temp_x_wspd']

# Filter out missing columns
feature_cols = [c for c in feature_cols if c in df.columns]

scaler = RobustScaler()
scaler.fit(train_df[feature_cols].fillna(0))  # fit on train only

# Apply scaler
train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

train_scaled[feature_cols] = scaler.transform(train_scaled[feature_cols].fillna(0))
val_scaled[feature_cols]   = scaler.transform(val_scaled[feature_cols].fillna(0))
test_scaled[feature_cols]  = scaler.transform(test_scaled[feature_cols].fillna(0))

# Save scaler and features list
joblib.dump(scaler, OUT_DIR / "robust_scaler.joblib")
with open(OUT_DIR / "feature_list.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

# Save CSVs (include target column and features)
cols_to_save = feature_cols + [target_col]
train_scaled[cols_to_save].to_csv(OUT_DIR / "train.csv")
val_scaled[cols_to_save].to_csv(OUT_DIR / "val.csv")
test_scaled[cols_to_save].to_csv(OUT_DIR / "test.csv")
print("Saved processed CSVs to", OUT_DIR)

# -----------------------
# Diagnostic plots
# -----------------------
plt.style.use('default')

# 1) Time series overview (full)
plt.figure(figsize=(14,4))
plt.plot(df.index, df[target_col], linewidth=0.5)
plt.title("Power demand (cleaned) — full period")
plt.xlabel("datetime")
plt.ylabel("Power demand (kW)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "power_time_series_full.png", dpi=150)
plt.close()

# 2) Missingness per column
plt.figure(figsize=(8,3))
sns.heatmap(df.isna().T, cbar=False)
plt.title("Missingness heatmap (rows=time, cols=features) — True=missing")
plt.savefig(PLOTS_DIR / "missingness_heatmap.png", dpi=150)
plt.close()

# 3) Seasonal decomposition (sample period)
try:
    # use a contiguous chunk for decomposition (avoid long series memory)
    decomp = seasonal_decompose(df[target_col].dropna().iloc[:288*60], model='additive', period=288)
    fig = decomp.plot()
    fig.set_size_inches(10,8)
    fig.suptitle("Seasonal decomposition (first ~60 days)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "seasonal_decompose.png", dpi=150)
    plt.close()
except Exception as e:
    print("Seasonal decomposition failed:", e)

# 4) ACF
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(8,4))
plot_acf(df[target_col].dropna().values, lags=600, ax=plt.gca())
plt.title("ACF of Power demand (lags up to ~600)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "acf_power.png", dpi=150)
plt.close()

# 5) Histogram + boxplot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df[target_col].dropna(), bins=100, kde=True)
plt.title("Distribution of Power demand")
plt.subplot(1,2,2)
sns.boxplot(x=df[target_col].dropna())
plt.title("Boxplot of Power demand")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "dist_box_power.png", dpi=150)
plt.close()

# 6) Correlation heatmap between features and target (small set)
corr_cols = ['power_imputed_flag','is_outlier','hour','dayofweek'] + [c for c in ['temp','dwpt','rhum','wspd','pres'] if c in df.columns] + [target_col]
corr = df[corr_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='vlag', center=0)
plt.title("Correlation matrix (subset)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "corr_subset.png", dpi=150)
plt.close()

print("Saved diagnostic plots to", PLOTS_DIR)
print("Done.")
