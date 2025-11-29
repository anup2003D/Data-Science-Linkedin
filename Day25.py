import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.DataFrame({
    "Age": [22, 35, 42, 28, 30],
    "Salary": [25000, 50000, 65000, 40000, 45000]
})

# Standardization
std_scaler = StandardScaler()
standardized = std_scaler.fit_transform(data)

# Normalization
minmax_scaler = MinMaxScaler()
normalized = minmax_scaler.fit_transform(data)

print("Standardized:\n", standardized)
print("\nNormalized:\n", normalized)
