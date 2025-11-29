import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset
data = pd.DataFrame({
    'age': [25, 30, np.nan, 40, 22],
    'income': [30000, 50000, 45000, np.nan, 28000],
    'city': ['A', 'B', 'A', 'C', 'B'],
    'score': [85, 90, 88, 92, 80]
})

X = data[['age', 'income', 'city']]
y = data['score']

# Numerical & categorical columns
num_cols = ['age', 'income']
cat_cols = ['city']

# Transformers
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

# Full pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
])

# Train & evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Test Score:", model.score(X_test, y_test))
