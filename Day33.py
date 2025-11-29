from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load dataset
data = load_wine()
X = data.data
y = data.target

# Build pipeline
model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=2000))
])

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold)

print("Cross-Validation Scores:", scores)
print("Average CV Score:", np.mean(scores))
