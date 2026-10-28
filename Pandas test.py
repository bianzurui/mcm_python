import pandas as pd
import matplotlib.pyplot as plt

# 读取外部csv文件
df = pd.read_csv("data.csv")

print("====原始数据====")
print(df)

# 打印所有列的名字
print("\n====所有列名====")
print(df.columns)

print("\n====基础统计信息====")
print(df.describe())

# 绘图，列名改成【序号】
plt.plot(df["序号"], df["数值"])
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Time Series Plot")
plt.savefig("read_csv_figure.png")
# 筛选：数值大于20的行
res = df[ df["数值"] > 20 ]
print("\n====数值大于20的数据====")
print(res)

# 统计筛选之后数据的平均值
print("\n筛选后数据的均值：", res["数值"].mean())
