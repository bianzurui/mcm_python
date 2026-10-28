from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
X=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([12,19,31,38,52,59,71,79,91,103])
kf=KFold(n_splits=5,shuffle=False)
mae_values=[]
for train_index,test_index in kf.split(X):
    X_train=X[train_index]
    X_test=X[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,y_pred)
    mae_values.append(mae)
    print("这一折的MAE：",round(mae,2))
    print("----------------")
    print("平均MAE：",round(sum(mae_values)/len(mae_values),2))