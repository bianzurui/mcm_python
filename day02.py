import pandas as pd
# 读取真实数据
df = pd.read_csv("MCM_2026_C_Data.csv")
# 1. 提取游客人数
visitor = df["Visitor"]
print("游客人数：")
print(visitor)
# 2. 同时提取日期和游客人数
new_df = df[["Date", "Visitor"]]
print("日期和游客人数：")
print(new_df)
# 3. 使用iloc提取第一行
print("第一行数据：")
print(df.iloc[0])
# 4. 使用iloc提取前三行
print("前三行数据：")
print(df.iloc[0:3])
# 5. 使用loc提取索引为0的数据
print("索引为0的数据：")
print(df.loc[0])
# 6. 筛选游客人数超过200的数据
high_visitors = df[df["Visitor"] > 200]
print("游客人数超过200的数据：")
print(high_visitors)
# 7. 筛选温度超过15度的数据
high_temperature = df[df["Temperature"] > 15]
print("温度超过15度的数据：")
print(high_temperature)
# 8. 同时满足两个条件
result = df[
    (df["Temperature"] > 15) &
    (df["Visitor"] > 200)
]
print("同时满足两个条件的数据：")
print(result)
# 9. 按游客人数从小到大排序
sorted_df = df.sort_values(by="Visitor")
print("游客人数升序排列：")
print(sorted_df)
# 10. 按游客人数从大到小排序
sorted_df = df.sort_values(by="Visitor", ascending=False)
print("游客人数降序排列：")
print(sorted_df)
# 11. 找出游客人数最多的5天
top5 = df.sort_values(
    by="Visitor",
    ascending=False
).head(5)
print("游客人数最多的5天：")
print(top5)
# 16. 检查重复数据数量
print("重复数据数量：")
print(df.duplicated().sum())
# 17. 查看重复数据
duplicate_data = df[df.duplicated()]
print("重复的数据：")
print(duplicate_data)
# 18. 删除重复数据
df_no_duplicates = df.drop_duplicates()
print("删除重复数据后的结果：")
print(df_no_duplicates)
# 19. 比较处理前后的数据规模
print("原始数据规模：", df.shape)
print("去重后的数据规模：", df_no_duplicates.shape)
# 20. 查看转换前的数据类型
print("转换前的数据类型：")
print(df.dtypes)
# 21. 将Date转换为日期类型
df["Date"] = pd.to_datetime(df["Date"])
# 22. 查看转换后的数据类型
print("转换后的数据类型：")
print(df.dtypes)
# 23. 提取年份和月份
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
print("提取年份和月份后的数据：")
print(df)







# 模拟练习
# Day2 综合数据清洗挑战

# 1. 读取数据
df = pd.read_csv("MCM_2026_C_Data.csv")

# 2. 查看数据规模
print("原始数据规模：")
print(df.shape)

# 3. 查看缺失值
print("缺失值数量：")
print(df.isnull().sum())

# 4. 删除重复数据
df = df.drop_duplicates()

# 5. 转换日期类型
df["Date"] = pd.to_datetime(df["Date"])

# 6. 填充温度缺失值
df["Temperature"] = df["Temperature"].fillna(
    df["Temperature"].mean()
)

# 7. 提取年份和月份
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# 8. 筛选游客人数超过200的数据
high_visitors = df[df["Visitor"] > 200]

print("游客人数超过200的数据：")
print(high_visitors)

# 9. 找出游客人数最多的5天
top5 = df.sort_values(
    by="Visitor",
    ascending=False
).head(5)

print("游客人数最多的5天：")
print(top5)
