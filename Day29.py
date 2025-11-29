from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

# Load dataset
X, y = load_iris(return_X_y=True)

# Correct train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline prevents data leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),    # scaling inside pipeline avoids leakage
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print("Test Accuracy:", accuracy)

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=kf)

print("CV Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())
