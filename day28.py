import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
X=np.array([[20],[22],[np.nan],[26],[28],[24],[30],[25],[21],[27]])
y=np.array([100,120,125,160,180,145,200,155,110,175])
pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="mean")),
    ("model",DecisionTreeRegressor(max_depth=3,random_state=42))
])
print(pipeline)
scores=cross_val_score(pipeline,X,y,cv=5,scoring="neg_mean_absolute_error")
mae_scores=-scores
print("每一折的MAE：")
print(mae_scores)
print("平均MAE：")
print(mae_scores.mean())