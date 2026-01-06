import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

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

# Plot PDP for selected features
features = ['MedInc', 'HouseAge', 'AveRooms']

PartialDependenceDisplay.from_estimator(model,X_train,features,grid_resolution=50)

plt.show()
