import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
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
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    ))
])
param_grid={
    "model__max_depth":[2,3,5,10]
    }
grid_search=GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error"
)
grid_search.fit(X,y)
print("最佳参数：",grid_search.best_params_)
print("最佳交叉验证MAE：",-grid_search.best_score_)
best_model=grid_search.best_estimator_
new_data=np.array([[23,0]])
prediction=best_model.predict(new_data)
print("预测游客人数：",prediction)