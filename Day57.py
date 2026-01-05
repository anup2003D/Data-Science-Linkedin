import pandas as pd
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Target and features
y = df['MedHouseVal']
X = df.drop(columns=['MedHouseVal'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test, check_additivity=False)


# Explain one prediction
shap.plots.waterfall(shap_values[0])
shap.plots.bar(shap_values)
