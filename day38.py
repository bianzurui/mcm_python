import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. 准备数据
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
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# 建立模型流程
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])
param_grid = {
    "model__max_depth": [2, 3, 5]
}
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error"
)
# 只使用训练集进行参数搜索
grid_search.fit(X_train, y_train)
# 获取最佳模型
best_model = grid_search.best_estimator_

# 使用测试集预测
y_pred = best_model.predict(X_test)

# 计算测试集MAE
mae = mean_absolute_error(y_test, y_pred)

print("最佳参数：", grid_search.best_params_)
print("最佳交叉验证MAE：", -grid_search.best_score_)
print("最终测试集MAE：", mae)
