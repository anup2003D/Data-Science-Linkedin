import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Sample data
data = {
    'Size_sqft': [850, 900, 1200, 1500, 1800, 2000, 2200, 2500],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 5],
    'Price': [150000, 160000, 200000, 250000, 280000, 310000, 350000, 400000]
}
df = pd.DataFrame(data)

# Splitting features & target
X = df[['Size_sqft', 'Bedrooms']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Predictions:", y_pred)
print("R² Score:", r2)
print("RMSE:", rmse)
