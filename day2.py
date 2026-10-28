import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("MCM_2026_C_Data.csv")

# 取出温度和游客数量
x = df["Temperature"]
y = df["Visitor"]

# 只保留两个数据都存在的行
mask = x.notna() & y.notna()

x = x[mask]
y = y[mask]

# 计算回归直线
slope, intercept = np.polyfit(x, y, 1)

# 生成预测值
y_pred = slope * x + intercept

# 画原始数据
plt.scatter(x, y)

# 画回归线
plt.plot(x, y_pred)

plt.xlabel("Temperature")
plt.ylabel("Visitor")
plt.title("Temperature vs Visitor")

plt.savefig("temperature_regression.png")