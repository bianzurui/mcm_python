import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
X=np.array([[20],[22],[np.nan],[26],[28],[24],[30],[25],[21],[27]])
y=np.array([100,120,125,160,180,145,200,155,110,175])
tree_pipeline=Pipeline([("imputer",SimpleImputer(strategy="mean")),
                        ("model",DecisionTreeRegressor(max_depth=3,random_state=42))
                        ])
forest_pipeline=Pipeline([("imputer",SimpleImputer(strategy="mean")),
                          ("model",RandomForestRegressor(
                              n_estimators=100,max_depth=3,random_state=42
                          ))
                          ])
tree_scores=cross_val_score(tree_pipeline,X,y,cv=5,scoring="neg_mean_absolute_error")
tree_mae=-tree_scores
print("决策树每一折的MAE：")
print(tree_mae)
print("决策树的平均MAE:")
print(tree_mae.mean())
froest_scores=cross_val_score(forest_pipeline,X,y,cv=5,scoring="neg_mean_absolute_error")
forest_mae=-froest_scores
print("随机森林每一折的MAE:")
print(forest_mae)
print("随机森林的平均MAE:")
print(forest_mae.mean())
print("模型比较结果：")
if tree_mae.mean() < forest_mae.mean():
    print("决策树的平均MAE更低:")
elif tree_mae.mean() > forest_mae.mean():
    print("随机森林的平均MAE更低")
else:
    print("两个模型的平均MAE相同")
print("开始测试不同的树深度：")
best_mae=float("inf")
best_depth=None
for depth in [2,3,5,10]:
    model=Pipeline([("imputer",SimpleImputer(strategy="mean")),("model",DecisionTreeRegressor(max_depth=depth,random_state=42))])
    scores=cross_val_score(model,X,y,cv=5,scoring="neg_mean_absolute_error")
    mae_scores=-scores
    print("树深度：",depth)
    print("平均MAE：",mae_scores.mean())
    if mae_scores.mean()<best_mae:
        best_mae=mae_scores.mean()
        best_depth=depth
print("最佳树深度：", best_depth)
print("最低平均MAE：", best_mae)