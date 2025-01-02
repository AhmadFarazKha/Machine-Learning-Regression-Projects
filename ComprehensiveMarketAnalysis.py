import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('house_prices.csv')

#Rename columns to match problem description
data.rename(columns={'size': 'store_size', 'price': 'sales'}, inplace=True)

# Create additional features with random data within specified ranges
np.random.seed(42)  # for reproducibility
data['num_employees'] = np.random.randint(10, 201, len(data))
data['marketing_budget'] = np.random.randint(1000, 100001, len(data))
data['competition_proximity'] = np.random.randint(0, 21, len(data))
data['years_in_operation'] = np.random.randint(1, 51, len(data))

#Create non-linear relationship (example)
data['sales'] = 5*data['store_size'] + 200*data['num_employees'] + 0.5*data['marketing_budget'] - 1000*data['competition_proximity'] + 1000*data['years_in_operation'] + np.random.normal(0, 100000, len(data))
data['sales'] = data['sales'].clip(10000, 1000000)

# Prepare data
X = data[['store_size', 'num_employees', 'marketing_budget', 'competition_proximity', 'years_in_operation']]
y = data['sales']

# Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models and Evaluation (using cross-validation - now correctly using X_scaled)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    mse_scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(mse_scores)
    results[name] = rmse_scores.mean()

print("Cross-Validation RMSE Results:")
for name, rmse in results.items():
    print(f"{name}: {rmse}")

# Train best model and make predictions (example with Linear Regression - now correctly using X_scaled)
best_model = LinearRegression()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nTest Set RMSE (Linear Regression): {rmse}")
print(f"Test Set R-squared (Linear Regression): {r2}")

#Confidence intervals (example)
predictions = best_model.predict(X_test)
errors = predictions - y_test
confidence_interval = np.percentile(errors, [2.5, 97.5])
print(f"95% Confidence interval for prediction errors: {confidence_interval}")

#Visualize predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()