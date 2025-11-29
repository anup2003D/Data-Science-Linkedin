import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataset
df = pd.DataFrame({
    'Age': [21, 23, 25, 27, 29, 31, 33, 35, 37, 39],
    'Salary': [25000, 27000, 30000, 35000, 40000, 45000, 52000, 60000, 68000, 75000],
    'Experience': [1, 2, 2, 3, 4, 5, 6, 7, 8, 10]
})

# 1️⃣ Histogram - Distribution of Age
sns.histplot(df['Age'], bins=5, kde=True)
plt.title("Age Distribution")
plt.show()

# 2️⃣ Scatter Plot - Relationship between Age and Salary
sns.scatterplot(x='Age', y='Salary', data=df)
plt.title("Age vs Salary")
plt.show()

# 3️⃣ Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
