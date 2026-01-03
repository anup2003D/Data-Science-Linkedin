import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("laptop_battery_health_usage.csv")

# Target and features
y = df['battery_health_percent']
X = df.drop(columns=['device_id', 'battery_health_percent'])

# One-hot encoding
X_encoded = pd.get_dummies(X,columns=['brand', 'os', 'usage_type', 'overheating_issues'],drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importance = pd.Series(rf.feature_importances_,index=X_encoded.columns).sort_values(ascending=False)

print("Top important features:")
print(importance.head(10))
