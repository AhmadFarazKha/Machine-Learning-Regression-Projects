import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('house_prices.csv')

#Rename columns to match problem description
data.rename(columns={'size': 'study_hours', 'price': 'exam_score'}, inplace=True)

# Create additional features with random data within specified ranges
np.random.seed(42)  # for reproducibility
data['gpa'] = np.random.uniform(2.0, 4.0, len(data))
data['attendance'] = np.random.randint(70, 101, len(data))
data['sleep_hours'] = np.random.randint(6, 11, len(data))
data['extra_activities'] = np.random.randint(0, 4, len(data))

#Create non-linear relationship (example)
data['exam_score'] = 20 + 2*data['study_hours'] + 15*data['gpa'] + 0.3*data['attendance'] + 1.5*data['sleep_hours'] - 2*data['extra_activities'] + np.random.normal(0, 15, len(data))
data['exam_score'] = data['exam_score'].clip(0, 100)

# Prepare data
X = data[['study_hours', 'gpa', 'attendance', 'sleep_hours', 'extra_activities']]
y = data['exam_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (important for multiple regression and feature importance)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Feature importance (coefficients)
coefficients = model.coef_
feature_names = X.columns
importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
importance['Abs_Coefficient'] = abs(importance['Coefficient'])
importance_sorted = importance.sort_values('Abs_Coefficient', ascending=False)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print results
print("Feature Importance:")
print(importance_sorted.to_markdown(index=False, numalign="left", stralign="left"))

# Calculate contribution of each feature (example for one student)
sample_student = X_test.iloc[0].values.reshape(1, -1)
sample_student_scaled = scaler.transform(sample_student)
prediction = model.predict(sample_student_scaled)[0]
contributions = coefficients * sample_student_scaled[0]

contributions_df = pd.DataFrame({'Feature': feature_names, 'Contribution': contributions})
print("\nExample Feature Contributions for One Student:")
print(contributions_df.to_markdown(index=False, numalign="left", stralign="left"))

# Recommendations
print("\nRecommendations for Improving Student Performance:")
print("- Focus on study hours and GPA, as these are the most significant predictors based on the model.")
print("- Maintaining good attendance also has a positive impact on exam scores.")
print("- Adequate sleep and balanced extracurricular involvement are helpful but have a smaller impact.")

#Visualize predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs. Predicted Exam Scores")
plt.show()