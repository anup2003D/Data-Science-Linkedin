from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Test Accuracy:", model.score(X_test, y_test))

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("CV Accuracy Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())
