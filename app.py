import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and dataset
lr = joblib.load('linear_model.pkl')
rf = joblib.load('rf_model.pkl')
df = pd.read_csv('BostonHousing_clean.csv')

st.title("🏠 AI-Powered House Price Prediction")

# Sidebar inputs
st.sidebar.header("User Preferences")
budget = st.sidebar.number_input("Enter your budget (in dollars)", min_value=0, step=1000)
rooms = st.sidebar.slider("Average rooms per house", min_value=3.0, max_value=9.0, value=6.0, step=0.1)
distance = st.sidebar.slider("Distance to employment centers", min_value=1.0, max_value=12.0, value=4.0, step=0.1)
crime_rate = st.sidebar.slider("Preferred maximum crime rate", min_value=float(df['crim'].min()), max_value=float(df['crim'].max()), value=float(df['crim'].median()), step=0.01)
salary = st.sidebar.number_input("Enter your monthly salary (in dollars)", min_value=0, step=1000)

# Prepare input for prediction
input_data = pd.DataFrame({
    'crim': [crime_rate],
    'zn': [df['zn'].median()],
    'indus': [df['indus'].median()],
    'chas': [0],
    'nox': [df['nox'].median()],
    'rm': [rooms],
    'age': [df['age'].median()],
    'dis': [distance],
    'rad': [df['rad'].median()],
    'tax': [df['tax'].median()],
    'ptratio': [df['ptratio'].median()],
    'b': [df['b'].median()],
    'lstat': [df['lstat'].median()]
})

# Predict using Random Forest
price = rf.predict(input_data)[0] * 1000  # Convert from $1000s to dollars

# Display prediction
st.subheader("Predicted House Price")
st.write(f"Estimated Price: ${price:,.2f}")

# Budget Check
st.subheader("Budget Check")
st.write(f"Your budget: ${budget:,.2f}")
if price <= budget:
    st.success("✅ This house fits your budget!")
else:
    st.error("❌ This house exceeds your budget!")

# Salary Affordability
affordable_limit = salary * 5
st.subheader("Salary-based Affordability")
st.write(f"Based on your salary, affordability limit: ${affordable_limit:,.2f}")
if price <= affordable_limit:
    st.info("💰 This house is affordable based on your salary!")
else:
    st.warning("⚠ This house may not be affordable.")

# Luxury and Near City Filters
st.subheader("Amenities Simulation")
is_luxury = rooms >= 6
is_near_city = distance <= 3
if is_luxury and is_near_city:
    # st.balloons()
    st.success("✨ This is a luxury house near the city!")
elif is_luxury:
    st.info("✨ This is a luxury house!")
elif is_near_city:
    st.info("🏙 This house is near the city!")

# Crime Rate Preference Check
st.subheader("Crime Rate Preference")
st.write(f"Your preferred max crime rate: {crime_rate:.2f}")
if crime_rate >= input_data['crim'][0]:
    st.success("✅ This house meets your crime rate preference!")
else:
    st.error("❌ This house exceeds your preferred crime rate!")

# Top 3 Similar Houses
st.subheader("Top 3 House Recommendations")
df['predicted'] = rf.predict(df.drop('medv', axis=1)) * 1000
df['room_diff'] = abs(df['rm'] - rooms)
df['dis_diff'] = abs(df['dis'] - distance)
df['crim_diff'] = abs(df['crim'] - crime_rate)
df['total_diff'] = df['room_diff'] + df['dis_diff'] + df['crim_diff']
top3 = df.nsmallest(3, 'total_diff')

for i, row in top3.iterrows():
    st.write(f"🏠 House {i+1}: Price = ${row['predicted']:,.2f}, Rooms = {row['rm']}, Distance = {row['dis']}, Crime Rate = {row['crim']:.2f}")

# Feature Importance Visualization
st.subheader("Feature Importance")
importances = rf.feature_importances_
features = df.drop(['medv', 'predicted', 'room_diff', 'dis_diff', 'crim_diff', 'total_diff'], axis=1).columns
fig, ax = plt.subplots()
sns.barplot(x=importances, y=features, ax=ax)
ax.set_title("Feature Importances")
st.pyplot(fig)






st.write("Developed by P.Nitish-AI powered house price prediction")