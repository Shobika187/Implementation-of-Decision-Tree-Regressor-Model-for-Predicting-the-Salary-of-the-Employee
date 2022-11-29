# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and get data info
2.check for null values
3.Map values for position column
4.Split the dataset into train and test set
5.Import decision tree regressor and fit it for data
6.Calculate MSE,R2 and y predict
## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Shobika P
RegisterNumber:  212221230096


import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data[["Salary"]]
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:
![image](https://user-images.githubusercontent.com/94508142/204555077-f13bbddb-338f-4d5b-bb3b-7fe12aa4fac4.png)

![image](https://user-images.githubusercontent.com/94508142/204555169-447944c1-3c37-4c0d-9d96-1750c5dff20a.png)

![image](https://user-images.githubusercontent.com/94508142/204555349-ca9a604e-ee51-49e7-9a4d-4ca789ae7ba9.png)

![image](https://user-images.githubusercontent.com/94508142/204555448-a1162cec-793e-490c-b8d6-5bf802798f0a.png)

![image](https://user-images.githubusercontent.com/94508142/204555682-d07c55b3-4c26-4260-8171-5fa8016dbc67.png)

![image](https://user-images.githubusercontent.com/94508142/204555818-a46ce47c-9ec5-4fb4-ab3f-43b42f36b5b9.png)

![image](https://user-images.githubusercontent.com/94508142/204556251-379e6100-5757-4d6c-8288-a60bd0b2ee1e.png)





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
