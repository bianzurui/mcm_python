import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
X = np.array([
    [20, 0],
    [22, 0],
    [np.nan, 1],
    [26, 0],
    [28, 0],
    [24, 1],
    [30, 0],
    [25, 1],
    [21, 0],
    [27, 1]
])
y = np.array([
    100, 120, 125, 160, 180,
    145, 200, 155, 110, 175
])
print("X的数据形状：", X.shape)
print("y的数据形状：", y.shape)
print("X的缺失值数量：", np.isnan(X).sum())
print("y的缺失值数量：", np.isnan(y).sum())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("训练集特征数量：", len(X_train))
print("测试集特征数量：", len(X_test))
print("训练集目标数量：", len(y_train))
print("测试集目标数量：", len(y_test))
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])
print(pipeline)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("测试集真实值：", y_test)
print("测试集预测值：", y_pred)
print("测试集MAE：", mae)