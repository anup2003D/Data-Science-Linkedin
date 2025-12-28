import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("laptop_battery_health_usage.csv")

# Define X and y
y = df['battery_health_percent']
X = df.drop(columns=['device_id', 'battery_health_percent'])

# One-hot encode categorical columns
X = pd.get_dummies(X,columns=['brand', 'os', 'usage_type', 'overheating_issues'],drop_first=True)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# High Bias Model (Underfitting)
lr = LinearRegression()
lr.fit(X_train, y_train)

train_lr = r2_score(y_train, lr.predict(X_train))
test_lr = r2_score(y_test, lr.predict(X_test))

print("Linear Regression (High Bias)")
print("Train R²:", train_lr)
print("Test R²:", test_lr)

# Balanced Model (Regularized)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

train_ridge = r2_score(y_train, ridge.predict(X_train))
test_ridge = r2_score(y_test, ridge.predict(X_test))

print("\nRidge Regression (Better Balance)")
print("Train R²:", train_ridge)
print("Test R²:", test_ridge)
