import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
data = {'Experience': [1, 2, 3, 4, 5, 6, 7, 8],
        'Salary': [25000, 28000, 35000, 40000, 45000, 52000, 60000, 65000]}
df = pd.DataFrame(data)

# Splitting data
X = df[['Experience']]
y = df['Salary']

# Training model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Visualizing the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression - Experience vs Salary')
plt.legend()
plt.show()

# Prediction for 5 years of experience
print("Predicted Salary for 5 years:", model.predict([[5]])[0])Day13.py