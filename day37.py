import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
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
y = np.array([
    100, 120, 125, 160, 180,
    145, 200, 155, 110, 175
])
model1 = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    ))
])
model2 = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ))
])
scores1 = cross_val_score(
    model1, X, y,
    cv=5,
    scoring="neg_mean_absolute_error"
)
scores2 = cross_val_score(
    model2, X, y,
    cv=5,
    scoring="neg_mean_absolute_error"
)
mae1 = -scores1
mae2 = -scores2
mean1 = np.mean(mae1)
mean2 = np.mean(mae2)
std1 = np.std(mae1)
std2 = np.std(mae2)
print("决策树每折MAE：", mae1)
print("决策树平均MAE：", mean1)
print("决策树标准差：", std1)

print("--------------------")

print("随机森林每折MAE：", mae2)
print("随机森林平均MAE：", mean2)
print("随机森林标准差：", std2)
