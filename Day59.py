import pandas as pd
import shap
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Features and target
y = df['MedHouseVal']
X = df.drop(columns=['MedHouseVal'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Error
errors = y_test - y_pred

# Create error DataFrame
error_df = X_test.copy()
error_df['Actual'] = y_test.values
error_df['Predicted'] = y_pred
error_df['Error'] = errors.values
error_df['Abs_Error'] = np.abs(errors.values)

# Top 5 worst predictions
worst_cases = error_df.sort_values(by='Abs_Error', ascending=False).head(5)
print(worst_cases[['Actual', 'Predicted', 'Error']])


# SHAP explainer
explainer = shap.Explainer(model, X_train)

# Disable strict additivity check (expected for tree models)
shap_values = explainer(X_test, check_additivity=False)

# Get positional index for SHAP
worst_pos = X_test.index.get_loc(worst_cases.index[0])

# Explain the failed prediction
shap.plots.waterfall(shap_values[worst_pos])


# Compare feature values for high-error cases
print(worst_cases[['MedInc', 'HouseAge', 'AveRooms', 'Latitude']])
