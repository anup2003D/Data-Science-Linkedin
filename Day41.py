import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer

# Load dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Introduce some missing values for practice
df.iloc[3:8, 2] = np.nan      # missing in feature 3
df.iloc[10:15, 5] = np.nan    # missing in feature 6

print("Missing Values Before:")
print(df.isnull().sum())

# 1. Mean imputation for numeric features
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\nMissing Values After Imputation:")
print(df_imputed.isnull().sum())