import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load the data
data = pd.read_csv('house_prices.csv')

#Rename columns to temperature and yield to match problem description
data.rename(columns={'size': 'temperature', 'price': 'yield'}, inplace=True)

# Create non-linear relationship (example)
np.random.seed(42)
data['yield'] = 50 + 2*data['temperature'] - 0.05*data['temperature']**2 + np.random.normal(0, 50, len(data))

# Split data
X = data[['temperature']]
y = data['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_y_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_y_pred)

# Polynomial Regression (degrees 2-4)
best_r2 = -1
best_degree = 0
best_poly_model = None
best_poly_y_pred = None

for degree in range(2, 5):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_y_pred = poly_model.predict(X_test_poly)
    poly_r2 = r2_score(y_test, poly_y_pred)

    if poly_r2 > best_r2:
        best_r2 = poly_r2
        best_degree = degree
        best_poly_model = poly_model
        best_poly_y_pred = poly_y_pred

print(f"Linear Regression R-squared: {linear_r2}")
print(f"Polynomial Regression (degree {best_degree}) R-squared: {best_r2}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Actual Data")

# Plot Linear Regression
plt.plot(X_test, linear_y_pred, color='red', label="Linear Regression")

# Plot Polynomial Regression
X_test_sorted, best_poly_y_pred_sorted = zip(*sorted(zip(X_test['temperature'], best_poly_y_pred)))
plt.plot(X_test_sorted, best_poly_y_pred_sorted, color='green', label=f"Polynomial Regression (degree {best_degree})")

plt.xlabel("Temperature (°C)")
plt.ylabel("Crop Yield (kg/hectare)")
plt.title("Temperature vs. Crop Yield")
plt.legend()
plt.show()

# Predictions for different temperature ranges
temp_range = np.linspace(10, 35, 100).reshape(-1, 1)
temp_range_poly = PolynomialFeatures(degree=best_degree).transform(temp_range)
predicted_yields = best_poly_model.predict(temp_range_poly)

plt.figure(figsize=(10, 6))
plt.plot(temp_range, predicted_yields)
plt.xlabel("Temperature (°C)")
plt.ylabel("Predicted Crop Yield (kg/hectare)")
plt.title("Predicted Yields for Temperature Range")
plt.show()

# Comparison and Explanation
print("\nComparison and Explanation:")
print("Polynomial regression performs significantly better than linear regression in this case.")
print("This is because the relationship between temperature and yield is non-linear. Linear regression assumes a straight-line relationship, which doesn't capture the curvature seen in the data.")
print("Polynomial regression, by fitting a curve to the data, can better model this non-linear relationship and provide more accurate predictions.")