# Project using Simple Linear Regression
# Goal: To fit a line to data about U.S. honey production trends and use that model to make predictions about future.

import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# 1) Check sample data
print(df.head())

# 2) Find mean of totalprod per year. Store in prod_per_year
prod_per_year = df.groupby('year').mean()[['totalprod']]
prod_per_year = prod_per_year.reset_index()

# 3) Create X (years) and reshape
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# 4) Create y (total production)
y = prod_per_year['totalprod']

# 5) Create scatterplot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.7, color='orange', s=100)
plt.xlabel('Year')
plt.ylabel('Total Honey Production')
plt.title('Honey Production Over Time')
plt.grid(True, alpha=0.3)

# 6) Create linear regression model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()

# 7) Fit the model
regr.fit(X, y)

# 8: Print slope and intercept
print(f"Slope (coefficient): {regr.coef_[0]:,.2f}")
print(f"Intercept: {regr.intercept_:,.2f}")

# 9) Predictions
y_predict = regr.predict(X)
print(f"First 5 predictions: {y_predict[:5]}\n")

# 10) Plot regression line
plt.plot(X, y_predict, color='blue', linewidth=2, label='Linear Regression')
plt.legend()

# 11) Create future years array
X_future = np.array(range(2013, 2051))
print(f"X_future before reshape: {X_future[:5]}... (showing first 5)")
X_future = X_future.reshape(-1, 1)
print(f"X_future shape after reshape: {X_future.shape}")
print("\n")

# 12) Predict future values
future_predict = regr.predict(X_future)
print(f"Prediction for 2020: {future_predict[7]:,.0f}")
print(f"Prediction for 2030: {future_predict[17]:,.0f}")
print(f"Prediction for 2040: {future_predict[27]:,.0f}")
print(f"Prediction for 2050: {future_predict[-1]:,.0f}")
print("\n")

# 13) Plot future predictions
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.7, color='orange', s=100, label='Historical Data')
plt.plot(X, y_predict, color='blue', linewidth=2, label='Fitted Line')
plt.plot(X_future, future_predict, color='red', linewidth=2, 
         linestyle='--', label='Future Predictions')
plt.xlabel('Year')
plt.ylabel('Total Honey Production')
plt.title('Honey Production: Historical & Future Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Add a horizontal line at y=0 to show if production goes negative
plt.axhline(y=0, color='black', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

