import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([10,15,25,35,50,68,85,110,150,200])

# Transform features into polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Predict
y_pred = model.predict(X_poly)

# Visualize
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Polynomial Fit')
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("Feature X")
plt.ylabel("Target Y")
plt.legend()
plt.show()

# Model parameters
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)