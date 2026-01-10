import pandas as pd
import shap
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Features and target
y = df['MedHouseVal']
X = df.drop(columns=['MedHouseVal'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model
baseline_model = RandomForestRegressor(n_estimators=100,random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
print("Baseline MAE:", mean_absolute_error(y_test, baseline_pred))

# Improved model with hyperparameter tuning
improved_model = RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=5,random_state=42)
improved_model.fit(X_train, y_train)
improved_pred = improved_model.predict(X_test)

print("Improved MAE:", mean_absolute_error(y_test, improved_pred))
print("MAE Improvement:", mean_absolute_error(y_test, improved_pred) - mean_absolute_error(y_test, baseline_pred))

# SHAP analysis for improved model
explainer = shap.Explainer(improved_model, X_train)
shap_values_improved = explainer(X_test, check_additivity=False)
shap.plots.bar(shap_values_improved)