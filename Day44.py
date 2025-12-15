from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Load dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Standardization
std_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    std_scaler.fit_transform(df),
    columns=df.columns
)

# Normalization
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(df),
    columns=df.columns
)

print("Original Data (first 5 rows):")
print(df.head())

print("\nStandardized Data (mean≈0, std≈1):")
print(df_standardized.head())

print("\nNormalized Data (range 0–1):")
print(df_normalized.head())
