import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1.Load the dataset
df=pd.read_csv("dataset/Employee.csv")
print(df.head())

# 2.Select the features and target
# Here inside the df we use double [] because, df expects a single thing so we cannot add like  df[ "age","city"] so we combine all inside a single [] and then pass that inside the df
X=df[[
    "Education","JoiningYear","City","PaymentTier","Age","Gender","EverBenched","ExperienceInCurrentDomain"
]]

y=df[
    "LeaveOrNot"
]

print("----------------------About the table X-----------------------------------")
print(X.info())

print("----------------------About the table Y-----------------------------------")
print(y.info())

#Encode the categorical(text) columns into numerical columns.
print("encoding the TEXT columns of X using #One-Hot Encoding#")


# 3. convert all the text columns into number using one hot encoding
X_encoded=pd.get_dummies( #df.get_dummies is not used because get_dummies is a pandas function and not a dataframe function.
    X,
    columns=["Education","City","Gender","EverBenched"],
    drop_first=True
)

print("")
print(X_encoded.isnull().sum())


# 4. split the data into training data and test data ( here 20% is test data and 80% is training data)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

# 5. train the ml model

#here we are using logistic regression ( it is not a regression model but it is an classification model , it will generally answer as yes/no, 1/0, true/false)
model = LogisticRegression(max_iter=1000) # max-iter means the model can try to learn in 1000 times and can also stop in middle if it right
model.fit(X_train, y_train)


# 6. evalute the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
