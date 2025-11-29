import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample data
data = {'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]}
df = pd.DataFrame(data)

# Split data
X = df[['Hours_Studied']]
y = df['Pass']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict probability
x_range = np.linspace(0, 10, 100).reshape(-1, 1)
y_prob = model.predict_proba(x_range)[:, 1]

# Plot sigmoid curve
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(x_range, y_prob, color='red', label='Sigmoid Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression - Sigmoid Function')
plt.legend()
plt.show()