import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
df=pd.read_csv("visitor_multivariate_regression_data.csv")
print(df)
print(df.columns)
X=df[
    [
        "Temperature",
        "Rain",
        "Holiday",
    ]
]
y=df["Visitor"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression() 
model.fit(X_train,y_train)
print("系数：",model.coef_)
print("截距：",model.intercept_)
print("训练集R2:",model.score(X_train,y_train))
print("测试集R2:",model.score(X_test,y_test))