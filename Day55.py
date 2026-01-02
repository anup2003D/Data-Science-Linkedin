import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("laptop_battery_health_usage.csv")

# Target and features
y = df['battery_health_percent']
X = df.drop(columns=['device_id', 'battery_health_percent'])

# One-hot encoding
X_encoded = pd.get_dummies(X,columns=['brand', 'os', 'usage_type', 'overheating_issues'],drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Decision Tree model
tree = DecisionTreeRegressor(max_depth=4,random_state=42)

# Train
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))