import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dataset
data = {'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
        'Attendance': [60,65,70,75,80,85,90,92,95,98],
        'Pass': [0,0,0,0,1,1,1,1,1,1]}
df = pd.DataFrame(data)

# Features & target
X = df[['Hours_Studied', 'Attendance']]
y = df['Pass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Plot tree
plt.figure(figsize=(10,6))
plot_tree(tree, feature_names=['Hours_Studied','Attendance'], class_names=['Fail','Pass'], filled=True)
plt.title("Decision Tree for Student Performance")
plt.show()

# Predict
print("Prediction for 7 hrs & 85% attendance:", tree.predict([[7, 85]])[0])
