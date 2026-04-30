import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load cleaned dataset
df = pd.read_csv('BostonHousing_clean.csv')

# Prepare features and target
X = df.drop('medv', axis=1)
y = df['medv']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R²: {r2_lr:.2f}")

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²: {r2_rf:.2f}")

# Compare models
print("\nModel Comparison:")
print(f"Linear Regression -> RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}")
print(f"Random Forest -> RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")

# Save models
joblib.dump(lr, 'linear_model.pkl')
joblib.dump(rf, 'rf_model.pkl')

print("Models saved as 'linear_model.pkl' and 'rf_model.pkl'")
