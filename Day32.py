from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Identify numeric columns
num_cols = X.columns.tolist()

# Pipeline for numeric features
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Full preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols)
    ]
)

# Final pipeline with polynomial features
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit & evaluate
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print("Test R² Score:", score)
