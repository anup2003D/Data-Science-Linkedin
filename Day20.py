import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
        'Score': [35,40,45,50,55,60,65,70,75,80]}
df = pd.DataFrame(data)

# Features & target
X = df[['Hours_Studied']]
y = df['Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Coefficients (slope):", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Visualize
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Best Fit Line')
plt.title("Linear Regression: Study Hours vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()
