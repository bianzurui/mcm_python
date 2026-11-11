import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
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
y = np.array([100, 120, 125, 160, 180,
              145, 200, 155, 110, 175])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("训练集数据：")
print(X_train)
print("测试集数据：")
print(X_test)
print("训练集数量：", len(X_train))
print("测试集数量：", len(X_test))
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", DecisionTreeRegressor(max_depth=3, random_state=42))
])
pipeline.fit(X_train,y_train)
print("模型训练完成！")
y_pred=pipeline.predict(X_test)
print("真实游客人数：")
print(y_test)
print("模型预测游客人数：")
print(y_pred)
test_mae=mean_absolute_error(y_test,y_pred)
print("测试集MAE：")
print(test_mae)
forest_pipeline=Pipeline([("imputer",SimpleImputer(strategy="mean")),
                          ("model",RandomForestRegressor(n_estimators=100,max_depth=3,random_state=42))])
forest_pipeline.fit(X_train,y_train)
forest_pred=forest_pipeline.predict(X_test)
forest_mae=mean_absolute_error(y_test,forest_pred)
print("随机森林预测结果：")
print(forest_pred)
print("随机森林测试集 MAE：")
print(forest_mae)
print("模型比较结果：")
if forest_mae < test_mae:
    print("随机森林表现更好")
elif forest_mae > test_mae:
    print("决策树表现更好")
else:
    print("两个模型表现相同")
improvement = (test_mae - forest_mae) / test_mae * 100
print("MAE下降比例：")
print(improvement, "%")
tree_scores=cross_val_score(pipeline,X_train,y_train,cv=5,scoring="neg_mean_absolute_error")
forest_scores=cross_val_score(forest_pipeline,X_train,y_train,cv=5,scoring="neg_mean_absolute_error")
print("决策树交叉验证MAE：")
print(-tree_scores.mean())
print("随机森林交叉验证MAE：")
print(-forest_scores.mean())
