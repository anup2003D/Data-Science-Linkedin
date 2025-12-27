import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#Load Dataset
df = pd.read_csv("laptop_battery_health_usage.csv")

#Define Features (X) and Target (y)
y = df['battery_health_percent']
X = df.drop(columns=['device_id', 'battery_health_percent'])

#One-Hot Encode Categorical Columns
X_encoded = pd.get_dummies(X,columns=['brand', 'os', 'usage_type', 'overheating_issues'],drop_first=True)

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

#Define Model
model = Ridge(alpha=1.0, max_iter=2000)

#Apply Cross-Validation
cv_scores = cross_val_score(model,X_scaled,y,cv=5,scoring='r2')

#Results
print("Cross-Validation R² Scores:", cv_scores)
print("Mean CV R² Score:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())
