import pandas as pd

# Load dataset
df = pd.read_csv("laptop_battery_health_usage.csv")

# Target and features
y = df['battery_health_percent']
X = df.drop(columns=['device_id', 'battery_health_percent'])

# One-hot encoding
X_encoded = pd.get_dummies(
    X,
    columns=['brand', 'os', 'usage_type', 'overheating_issues'],
    drop_first=True
)

# Correlation with target
correlation = X_encoded.corrwith(y)

# Sort by importance
correlation_sorted = correlation.abs().sort_values(ascending=False)

print(correlation_sorted.head(10))

threshold = 0.1
selected_features = correlation_sorted[correlation_sorted > threshold].index

X_selected = X_encoded[selected_features]

print("Before:", X_encoded.shape[1])
print("After:", X_selected.shape[1])
