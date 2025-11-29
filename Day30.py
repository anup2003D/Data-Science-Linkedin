from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
X, y = load_iris(return_X_y=True)

# ----- Filter Method -----
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)
print("Filter Method - Selected Feature Indices:", selector.get_support(indices=True))

# ----- Wrapper Method (RFE) -----
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X, y)
print("Wrapper Method (RFE) - Selected Features:", rfe.get_support(indices=True))

# ----- Embedded Method (Random Forest) -----
forest = RandomForestClassifier()
forest.fit(X, y)
importance = pd.DataFrame({
    "feature": range(X.shape[1]),
    "importance": forest.feature_importances_
})
print("\nEmbedded Method (Random Forest) – Feature Importances:")
print(importance.sort_values(by="importance", ascending=False))
