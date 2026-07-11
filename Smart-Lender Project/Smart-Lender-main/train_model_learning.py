import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Read the dataset
data = pd.read_csv("dataset/loan_prediction.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Remove Loan_ID column
data.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values in categorical columns
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Fill missing values in numerical columns
numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in numerical_columns:
    data[col] = data[col].fillna(data[col].mean())

# Check for missing values
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Display cleaned dataset
print("\nCleaned Dataset:")
print(data.head())
# -----------------------------
# Encode text columns
# -----------------------------

# Create LabelEncoder object
encoder = LabelEncoder()

# List of text columns
text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

# Convert text into numbers
for col in text_columns:
    data[col] = encoder.fit_transform(data[col])

# Display encoded dataset
print("\nEncoded Dataset:")
print(data.head())
# =====================================
# Create Features (X) and Target (y)
# =====================================

# Features (Input)
X = data.drop("Loan_Status", axis=1)

# Target (Output)
y = data["Loan_Status"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
# =====================================
# Split Dataset
# =====================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
# =====================================
# Train Logistic Regression Model
# =====================================

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")
# =====================================
# Test the Model
# =====================================

from sklearn.metrics import accuracy_score

# Predict using test data
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
# =====================================
# Save the Trained Model
# =====================================

import joblib

joblib.dump(model, "models/model.pkl")

print("\n✅ Model saved successfully!")







































