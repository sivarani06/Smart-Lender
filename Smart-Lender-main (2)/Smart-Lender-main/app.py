from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read values from HTML form
    gender = request.form["Gender"]
    married = request.form["Married"]
    dependents = request.form["Dependents"]
    education = request.form["Education"]
    self_employed = request.form["Self_Employed"]

    applicant_income = float(request.form["ApplicantIncome"])
    coapplicant_income = float(request.form["CoapplicantIncome"])
    loan_amount = float(request.form["LoanAmount"])
    loan_term = float(request.form["Loan_Amount_Term"])
    credit_history = float(request.form["Credit_History"])

    property_area = request.form["Property_Area"]

    # Manual Encoding (same as training)
    gender = 1 if gender == "Male" else 0

    married = 1 if married == "Yes" else 0

    dependents = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3+": 3
    }[dependents]

    education = 0 if education == "Graduate" else 1

    self_employed = 1 if self_employed == "Yes" else 0

    property_area = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]

    # Create DataFrame
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # Prediction
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)