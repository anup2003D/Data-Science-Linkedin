import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

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

# Model
model = LinearRegression()

# RFE - select top 5 features
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_encoded, y)

# Selected features
selected_features = X_encoded.columns[rfe.support_]

print("Selected Features:")
print(selected_features)