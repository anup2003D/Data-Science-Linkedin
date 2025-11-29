import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create sample dataset
np.random.seed(0)
X = np.random.randn(100, 5)
y = 3*X[:,0] + 1.5*X[:,1] - 2*X[:,2] + np.random.randn(100)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Compare coefficients
print("Ridge Coefficients:", ridge.coef_)
print("Lasso Coefficients:", lasso.coef_)