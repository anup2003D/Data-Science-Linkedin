import pickle
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
data = load_wine()
X, y = data.data, data.target

# Pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=2000))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# ----- Save using Pickle -----
with open("model_pickle.pkl", "wb") as f:
    pickle.dump(model, f)

# ----- Load using Pickle -----
with open("model_pickle.pkl", "rb") as f:
    loaded_pickle_model = pickle.load(f)

print("Pickle Model Test Score:", loaded_pickle_model.score(X_test, y_test))

# ----- Save using Joblib -----
joblib.dump(model, "model_joblib.pkl")

# ----- Load using Joblib -----
loaded_joblib_model = joblib.load("model_joblib.pkl")

print("Joblib Model Test Score:", loaded_joblib_model.score(X_test, y_test))
