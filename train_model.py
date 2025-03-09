import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
import sys

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from config import MODEL_PATH

# Load the training data
data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_data.csv"))

# Prepare the features (X) and target (y)
# Ensure that 'skill_match_count' and 'experience' columns exist in your CSV
X = data[['skill_match_count', 'experience']]
y = data['suitability_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
