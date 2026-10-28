from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
X=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([12,19,31,38,52,59,71,79,91,103])
kf=KFold(n_splits=5,shuffle=False)
mae_A=[]
mae_B=[]
for train_index,test_index in kf.split(X):
    X_train=X[train_index]
    X_test=X[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    model_A=LinearRegression()
    model_B=DecisionTreeRegressor(max_depth=2)
    model_A.fit(X_train,y_train)
    model_B.fit(X_train,y_train)
    pred_A=model_A.predict(X_test)
    pred_B=model_B.predict(X_test)
    mae_A.append(mean_absolute_error(y_test,pred_A))
    mae_B.append(mean_absolute_error(y_test,pred_B))
print("模型A的每折MAE：",[round(x,2) for x in mae_A])
print("模型B的每折MAE：",[round(x,2) for x in mae_B])
avg_A=sum(mae_A)/len(mae_B)
avg_B=sum(mae_B)/len(mae_B)
print("模型A的平均MAE：",round(avg_A,2))
print("模型B的平均MAE：",round(avg_B,2))
if avg_A < avg_B:
    print("模型A的表现更好")
elif avg_A > avg_B:
    print("模型B的表现更好")
else:
    print("两个模型的平均MAE相同")    