import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox, boxcox_normmax

# 1. Load the dataset
data_path = 'OrangeJuiceX25.csv'
df = pd.read_csv(data_path)

# Drop the unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. Check for Missing Values
print("=== Missing Values ===")
print(df.isnull().sum())

# 3. Outlier Detection for 'sales' using IQR
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create a dummy variable 'sales_outlier'
df['sales_outlier'] = ((df['sales'] < lower_bound) | (df['sales'] > upper_bound)).astype(int)

print(f"\nTotal Outliers in 'sales': {df['sales_outlier'].sum()} out of {len(df)} rows")

# Plot sales distribution and mark outlier cutoffs
plt.figure(figsize=(8, 5))
plt.hist(df['sales'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(lower_bound, color='red', linestyle='--', linewidth=2, label='Lower IQR Bound')
plt.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label='Upper IQR Bound')
plt.title("Sales Distribution with Outlier Cutoffs")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Seasonality Analysis: Aggregate Weekly Sales
# ------------------------------------------------

# Sum total sales across all store-brand combinations by week
weekly_sales = df.groupby('week')['sales'].sum().sort_index()

# Create a date index by assuming week 1 corresponds to an arbitrary date (e.g., 2000-01-02 as a Sunday)
# and then generating consecutive weekly dates
start_date = '2000-01-02'
date_index = pd.date_range(start=start_date, periods=len(weekly_sales), freq='W-SUN')
series = pd.Series(weekly_sales.values, index=date_index)

# STL Decomposition (assuming annual seasonality ~52 weeks)
stl = STL(series, period=52, robust=True)
res = stl.fit()

# Plot STL components
stl_fig = res.plot()
stl_fig.tight_layout()
plt.show()

# 5. ACF and PACF Plots for Weekly Aggregated Sales
# -------------------------------------------------
n = len(series)
max_lag = int(min(52, (n // 2) - 1))

# Autocorrelation Function (ACF)
plt.figure(figsize=(10, 5))
plot_acf(series, lags=max_lag, zero=False)
plt.title("Autocorrelation (ACF) of Weekly Total Sales")
plt.xlabel("Lag (weeks)")
plt.ylabel("ACF")
plt.tight_layout()
plt.show()

# Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 5))
plot_pacf(series, lags=max_lag, zero=False, method='ywm')
plt.title("Partial Autocorrelation (PACF) of Weekly Total Sales")
plt.xlabel("Lag (weeks)")
plt.ylabel("PACF")
plt.tight_layout()
plt.show()

# 6. Variance-Stabilization: Box–Cox Transformation
# -------------------------------------------------
# Sales must be positive; add 1 to avoid zeros
sales_values = df['sales'].values + 1

# Find optimal lambda for Box–Cox
lambda_opt = boxcox_normmax(sales_values, method='mle')
print(f"\nOptimal Box–Cox Lambda: {lambda_opt:.3f}")

# Transform the sales
sales_boxcox = boxcox(sales_values, lmbda=lambda_opt)

# Plot original vs. transformed distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['sales'], bins=50, alpha=0.7, edgecolor='black')
plt.title("Original Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(sales_boxcox, bins=50, alpha=0.7, edgecolor='black')
plt.title(f"Box–Cox Transformed Sales (λ = {lambda_opt:.3f})")
plt.xlabel("Transformed Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
