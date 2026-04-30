# 🏠 AI-Powered House Price Prediction System

## 📖 Project Overview
This project is an AI-driven system that predicts house prices using the **Boston Housing dataset**. It helps users find houses within their budget based on preferences such as the number of rooms, distance to the city, and crime rate. The solution also compares two machine learning models — **Linear Regression** and **Random Forest Regressor** — and offers explainability through feature importance.

---

## 🎯 Problem Statement
- Predict the median value of houses (`MEDV`) using features like crime rate (`CRIM`), rooms (`RM`), distance (`DIS`), etc.
- Compare model performance using metrics like RMSE and R².
- Build a user-friendly application where users can input preferences and get recommendations.
- Implement filters for affordability, luxury, and proximity to the city.

---

## 📂 Dataset
- **Source:** Boston Housing dataset.
- **Features include:**
  - `CRIM`: Crime rate per capita
  - `RM`: Average number of rooms per dwelling
  - `DIS`: Weighted distances to employment centers
  - `LSTAT`: Percentage of lower status population
  - `MEDV`: Median house price (target variable)
  - ...and other factors influencing house prices.

---

## ✅ Key Features
- **Data Exploration & Visualization:**
  - Correlation heatmap to understand relationships between variables.
  - Pairplot for visualizing trends among features.
  
- **Model Development:**
  - Linear Regression as a baseline model.
  - Random Forest Regressor for improved accuracy.
  - Performance measured using RMSE and R².

- **Interactive App with Streamlit:**
  - User inputs preferences such as budget, rooms, distance, and crime rate.
  - Displays estimated price and top 3 recommended houses.
  - Filters based on salary, luxury, and proximity to the city.

- **Explainability:**
  - Shows feature importance to help users understand how the prediction is made.

---

## 📊 Model Performance

| Model             | RMSE   | R²   |
|-----------------|------|----|
| Linear Regression | 4.95 | 0.67 |
| Random Forest     | 2.84 | 0.89 |

✔ Random Forest significantly outperforms the Linear Regression model, explaining 89% of price variance.

---

## 🛠 Tools & Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit
- Pickle (for saving models)

---

## 📥 How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd house-price-prediction

2. Set up a Python virtual environment:

    python -m venv venv
    venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux

3. Install required libraries:

    pip install -r requirements.txt

4. Run the model training script:

    python model_training.py

5. Launch the Streamlit app:

    streamlit run app.py




📌 Future Enhancements

Add more features like neighborhood data or property age.

Deploy the app on a cloud platform.

Improve explainability using SHAP or LIME.

Implement real-time data updates.



🎉 Conclusion

This project demonstrates the application of machine learning techniques in real-world scenarios like housing markets. It emphasizes data exploration, model evaluation, and user-centric design, providing an end-to-end solution from dataset analysis to interactive recommendations.
