import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv('BostonHousing_clean.csv')

# Print summary statistics
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for a few features
sns.pairplot(df[['rm', 'dis', 'lstat', 'medv']])
plt.show()
