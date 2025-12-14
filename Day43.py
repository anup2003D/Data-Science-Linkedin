import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Sample dataset
df = pd.DataFrame({
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Kolkata'],
    'Education': ['Low', 'Medium', 'High', 'Medium']
})

print("Original Data:")
print(df)

# 1. Label Encoding
le = LabelEncoder()
df['City_Label'] = le.fit_transform(df['City'])

# 2. One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['City'])

# 3. Ordinal Encoding
ordinal_enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Education_Ordinal'] = ordinal_enc.fit_transform(df[['Education']])

print("\nLabel Encoded:")
print(df[['City', 'City_Label']])

print("\nOne-Hot Encoded:")
print(df_onehot)

print("\nOrdinal Encoded:")
print(df[['Education', 'Education_Ordinal']])
