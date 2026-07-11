# ==========================================
# SMART LENDER - MODEL TRAINING
# ==========================================

# Import Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# Read Dataset
# ==========================================

data = pd.read_csv("dataset/loan_prediction.csv")

# ==========================================
# Remove Unnecessary Columns
# ==========================================

data.drop("Loan_ID", axis=1, inplace=True)

# ==========================================
# Handle Missing Values
# ==========================================

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in numerical_columns:
    data[col] = data[col].fillna(data[col].mean())

# ==========================================
# Encode Text Columns
# ==========================================

# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
# ==========================================
# Encode Text Columns
# ==========================================

encoders = {}

text_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# ==========================================
# Create Features and Target
# ==========================================

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================
# Evaluate Model
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy : {accuracy:.2%}")

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")
joblib.dump(encoders, "models/encoders.pkl")