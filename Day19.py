import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dataset
data = {'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
        'Attendance': [60,65,70,75,80,85,90,92,95,98],
        'Assignments_Submitted': [2,3,4,5,6,7,8,9,10,10],
        'Pass': [0,0,0,0,1,1,1,1,1,1]}
df = pd.DataFrame(data)

# Features & target
X = df[['Hours_Studied', 'Attendance', 'Assignments_Submitted']]
y = df['Pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Feature Importances
importances = rf.feature_importances_
features = X.columns

# Visualize
plt.figure(figsize=(7,4))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()

# Display ranking
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")
