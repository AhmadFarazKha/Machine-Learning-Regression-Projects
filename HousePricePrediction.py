import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Load and Prepare Data ---
# Assuming your data is in a CSV file named 'house_prices.csv'
# with columns 'size' and 'price'
data= pd.read_csv('house_prices.csv')

# Split data into features (X) and target (y)
X = data[['size']]  # Feature: House size
y = data['price']   # Target: House price

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Build and Train the Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate Model Performance ---
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print(f"R-squared: {r_squared}")

# --- Make Predictions for New Houses ---
new_houses = [[1200], [2300], [3400]]
predicted_prices = model.predict(new_houses)
print(f"Predicted Prices: {predicted_prices}")

# --- Pakistani Real-Life Example ---
print("\nPakistani Example:")
print("Imagine a real estate company in Lahore wants to predict the price of apartments in DHA based on their size in square feet.")
print("This model could help them understand the relationship between size and price, and make informed decisions about pricing and property investments.")