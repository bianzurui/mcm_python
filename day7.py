import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error,mean_squared_error
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
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2=model.score(X_test,y_test)
print("测试集MAE:",mae)
print("测试集RMSE:",rmse)
print("测试集R²：",r2)
print("系数：",model.coef_)
print("截距：",model.intercept_)
print("训练集R2:",model.score(X_train,y_train))
print("测试集R2:",model.score(X_test,y_test))