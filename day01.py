import numpy as np
import pandas as pd
# 1. 创建 NumPy 数组
visitors = np.array([1200, 950, 1500, 800])
print("NumPy数组：")
print(visitors)
# 2. 创建 Pandas Series
s = pd.Series([1200, 950, 1500, 800])
print("Pandas Series：")
print(s)
# 3. 创建 Pandas DataFrame
data = {
    "Temperature": [25, 22, 28, 20],
    "Rain": [0, 5, 0, 10],
    "Visitor": [1200, 950, 1500, 800]
}
df = pd.DataFrame(data)
print("Pandas DataFrame：")
print(df)
# 4. 读取真实美赛数据
df = pd.read_csv("MCM_2026_C_Data.csv")

# 5. 查看数据
print("数据前5行：")
print(df.head())

# 6. 查看数据规模
print("数据规模：")
print(df.shape)
# 7. 检查数据基本信息
print("数据基本信息：")
df.info()

# 8. 检查每列缺失值
print("每列缺失值数量：")
print(df.isnull().sum())
# 9. 描述性统计
print("数据描述性统计：")
print(df.describe())
# 10. 筛选游客人数超过200的数据
high_visitors = df[df["Visitor"] > 200]

print("游客人数超过200的数据：")
print(high_visitors)