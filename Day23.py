import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create dataset
np.random.seed(0)
X = np.sort(np.random.rand(15, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(15) * 0.3  # noisy sine wave

# Train models with different degrees
degrees = [1, 4, 10]
plt.figure(figsize=(10, 6))

for i, degree in enumerate(degrees, 1):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    plt.subplot(1, 3, i)
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.title(f"Degree = {degree}")
    plt.xlabel("X")
    plt.ylabel("y")

plt.suptitle("Underfitting vs Good Fit vs Overfitting", fontsize=14)
plt.tight_layout()
plt.show()