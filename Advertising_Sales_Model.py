# Step 1: Import required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 2: Load dataset from URL
url = "https://www.statlearning.com/s/Advertising.csv"
dataset = pd.read_csv(url)

# Step 3: Check and print column names
print("Column names in dataset:", dataset.columns)

# Define features (independent variables) and target (dependent variable)
X = dataset[['TV', 'radio', 'newspaper']]  # Adjusted column names based on dataset
y = dataset['sales']  # Target: Sales outcome

# Print dataset info for verification
print(dataset.head())
print(dataset.info())
print(dataset.describe())

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Step 7: Plot prediction accuracy
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')

plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Prediction Accuracy of Multiple Regression Model')
plt.show()
