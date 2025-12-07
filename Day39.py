from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load dataset
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=2000))
])

# 1️⃣ Cross-validation on the base model
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
print("Base CV Scores:", cv_scores)
print("Base Average Score:", np.mean(cv_scores))

# 2️⃣ Hyperparameter tuning with GridSearch
param_grid = {
    'logreg__C': [0.1, 1, 5, 10],
    'logreg__solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# 3️⃣ Final evaluation on test data
test_score = grid.score(X_test, y_test)
print("Final Test Score:", test_score)