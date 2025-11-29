from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
data = load_wine()
X, y = data.data, data.target

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=2000))
])

# Hyperparameter grid
param_grid = {
    'logreg__C': [0.1, 1, 5, 10],
    'logreg__solver': ['lbfgs', 'liblinear'],
    'logreg__penalty': ['l2']
}

# Grid search
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
