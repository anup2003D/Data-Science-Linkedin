import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = {'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
        'Attendance': [60,65,70,75,80,85,90,92,95,98],
        'Pass': [0,0,0,0,1,1,1,1,1,1]}
df = pd.DataFrame(data)

# Features & target
X = df[['Hours_Studied', 'Attendance']]
y = df['Pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Prediction for [7 hrs, 85% attendance]:", rf.predict([[7,85]])[0])
