import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
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
    X, y,
    test_size=0.2,
    random_state=42
)

print("训练集数量：", len(X_train))
print("测试集数量：", len(X_test))
pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="mean")),
    ("model",RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ))
])
param_grid={
    "model__n_estimators":[50,100,200],
    "model__max_depth":[2,3,5]
}
grid_search=GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error"
)
grid_search.fit(X_train, y_train)
print("最佳参数：", grid_search.best_params_)
print("最佳交叉验证MAE：", -grid_search.best_score_)
best_model=grid_search.best_estimator_
# 提取随机森林的特征重要性
importance = best_model.named_steps["model"].feature_importances_

# 特征名称
feature_names = ["温度", "是否周末"]

# 输出结果
for name, value in zip(feature_names, importance):
    print(name, "的重要性：", value)
y_pred = best_model.predict(X_test)
residuals=y_test-y_pred
print("预测残差：",residuals)
mean_residual=np.mean(residuals)
print("平均残差：",mean_residual)
# 计算MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
# 计算RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 计算R²
r2 = r2_score(y_test, y_pred)

print("测试集MAE：", mae)
print("测试集RMSE：", rmse)
print("测试集R²：", r2)
print("测试集真实人数：", y_test)
print("测试集预测人数：", y_pred)
new_data=np.array([
    [23,0]
])
prediction=best_model.predict(new_data)
print("随机森林预测游客人数：",prediction)