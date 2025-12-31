import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

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

# Scale features (important for Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Lasso model
lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_scaled, y)

# Extract selected features
coef = pd.Series(lasso.coef_, index=X_encoded.columns)
selected_features = coef[coef != 0].index

print("Selected Features:")
print(selected_features)
