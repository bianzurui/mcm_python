import pandas as pd
df=pd.read_csv("MCM_2026_C_Data.csv")
print(df.isna().sum())
df_clean=df.dropna()
print("清洗前的数据",len(df))
print("清洗后的数据",len(df_clean))
df_fill = df.copy()

df_fill["Temperature"] = df_fill["Temperature"].fillna(
    df_fill["Temperature"].mean()
)

df_fill["Visitor"] = df_fill["Visitor"].fillna(
    df_fill["Visitor"].mean()
)

print(df_fill)
# 删除缺失值
df_drop = df.dropna()

# 用平均值填补缺失值
df_fill = df.copy()
df_fill["Temperature"] = df_fill["Temperature"].fillna(df_fill["Temperature"].mean())
df_fill["Visitor"] = df_fill["Visitor"].fillna(df_fill["Visitor"].mean())

# 分别计算相关系数
corr_drop = df_drop["Temperature"].corr(df_drop["Visitor"])
corr_fill = df_fill["Temperature"].corr(df_fill["Visitor"])

print("删除缺失值后的相关系数：", corr_drop)
print("填补缺失值后的相关系数：", corr_fill)