import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
X=np.array([[20],[22],[np.nan],[26],[28]])
y=np.array([100,120,125,160,180])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("训练集温度：")
print(X_train)
print("测试集温度")
print(X_test)
imputer=SimpleImputer(strategy="mean")
X_train_filled=imputer.fit_transform(X_train)
print("填充后的训练集温度：")
print(X_train_filled)
X_test_filled=imputer.transform(X_test)
print("填充后的测试集温度：")
print(X_test_filled)
model=DecisionTreeRegressor(max_depth=5,random_state=42)
model.fit(X_train_filled,y_train)
pred=model.predict(X_test_filled)
print("预测游客人数：")
print(pred)
mae=mean_absolute_error(y_test,pred)
print("测试集MAE:")
print(mae)