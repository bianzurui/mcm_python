import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("visitor_multivariate_regression_data.csv")
x=df["Temperature"]
y=df["Visitor"]
plt.scatter(x,y)
slope,intercept=np.polyfit(x,y,1)
Temperature=26
Visitor_pred=slope*Temperature+intercept
print("预测游客数量：",Visitor_pred)
y_pred=slope*x+intercept
print("斜率：",slope)
print("截距：",intercept)
plt.plot(x,y_pred)
plt.xlabel("Temperature")
plt.ylabel("Visitor")
plt.title("Temperature vs Visitor")
r = np.corrcoef(x, y)[0, 1]
print("相关系数 r：", r)
r2=r**2
print("R²:",r2)
plt.show()