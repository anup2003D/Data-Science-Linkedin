import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("laptop_battery_health_usage.csv")

# Target and features
y = df['battery_health_percent']
X = df.drop(columns=['device_id', 'battery_health_percent'])

# One-hot encode categorical features
X_encoded = pd.get_dummies(X,columns=['brand', 'os', 'usage_type', 'overheating_issues'],drop_first=True)

# Scale features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original feature count:", X_encoded.shape[1])
print("Reduced feature count:", X_pca.shape[1])

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", pca.explained_variance_ratio_.sum())

# Get PCA loadings
loadings = pd.DataFrame(pca.components_,columns=X_encoded.columns,index=[f"PC{i+1}" for i in range(pca.n_components_)])

print(loadings)

for pc in loadings.index:
    print(f"\nTop features contributing to {pc}:")
    print(loadings.loc[pc].abs().sort_values(ascending=False).head(5))
