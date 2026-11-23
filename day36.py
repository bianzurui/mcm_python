import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ))
])
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)
mae_scores = -scores
print("每一折的MAE：", mae_scores)
print("平均MAE：", np.mean(mae_scores))
