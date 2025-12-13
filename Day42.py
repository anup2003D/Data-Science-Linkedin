import pandas as pd
from sklearn.datasets import load_wine

# Load dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Select a feature to demonstrate outlier detection
feature = "color_intensity"

Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

print("Number of Outliers:", len(outliers))
print("\nOutlier Rows:")
print(outliers.head())

# Removing outliers
df_cleaned = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

print("\nShape Before:", df.shape)
print("Shape After Removing Outliers:", df_cleaned.shape)
