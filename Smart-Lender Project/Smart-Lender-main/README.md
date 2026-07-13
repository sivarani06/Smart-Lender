🏦 Smart Lender – Loan Approval Prediction System

## 📖 Overview

Smart Lender is a Machine Learning-powered web application that predicts loan approval based on applicant information. The system analyzes features such as gender, marital status, education, employment status, income, loan amount, loan term, credit history, and property area using classification algorithms including Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and XGBoost. The best-performing model is integrated with a Flask web application, enabling users to receive real-time loan approval predictions through a simple and responsive interface.

---

## 📂 Project Structure

```text
Smart-Lender/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── loan_prediction.csv
│
├── models/
│   ├── model.pkl
│   └── encoders.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── style.css
```

---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Joblib
- HTML5
- CSS3
- Visual Studio Code
- Git & GitHub

---

## 🚀 How to Run

```bash
git clone https://github.com/sivarani06/Smart-Lender.git

cd Smart-Lender

pip install -r requirements.txt

python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 👨‍💻 Author

**Attim Siva Rani**

GitHub: https://github.com/sivarani06/Smart-Lender.git
