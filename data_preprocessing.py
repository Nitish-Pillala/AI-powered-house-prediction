import pandas as pd

# Load dataset
df = pd.read_csv('BostonHousing.csv')

# Fill missing values (if any) in 'rm' column with median
df['rm'] = df['rm'].fillna(df['rm'].median())

# Save cleaned dataset
df.to_csv('BostonHousing_clean.csv', index=False)

print("Data cleaned and saved as 'BostonHousing_clean.csv'")
