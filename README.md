# Project Overview
This project implements a Logistic Regression machine learning model to predict whether an employee is likely to leave the organization or not based on various personal, professional, and organizational factors.
Employee attrition is a major concern for organizations, and predictive analytics can help HR teams make data-driven decisions to reduce turnover.

# Objective
To build a binary classification model that predicts:
1 → Employee will leave
0 → Employee will stay
Using features such as education, experience, city, age, gender, and work history.

# Machine Learning Concept Used
Logistic Regression
A classification algorithm, not a regression algorithm
Used when the target variable is binary
Outputs probabilities which are mapped to class labels (0 or 1)

# Technologies & Libraries Used
Python
NumPy
Pandas
Scikit-learn

# Final Output Example
Accuracy: 0.7443609022556391
              precision    recall  f1-score   support

           0       0.76      0.90      0.82       610
           1       0.70      0.45      0.55       321

    accuracy                           0.74       931
   macro avg       0.73      0.68      0.69       931
weighted avg       0.74      0.74      0.73       931

# Future Improvements
Feature scaling using StandardScaler
Try other classifiers (Random Forest, SVM, XGBoost)
Hyperparameter tuning using GridSearchCV
Deploy the model using Flask / FastAPI
Add visualization (confusion matrix, ROC curve)

# How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the script
python logistic_regression_employee.py