import pandas as pd
import matplotlib.pyplot as plt
# 1. 读取数据
df = pd.read_csv("MCM_2026_C_Data.csv")
# 2. 转换日期格式
df["Date"] = pd.to_datetime(df["Date"])
# 3. 按日期排序
df = df.sort_values(by="Date")
# 4. 绘制游客人数折线图
plt.plot(df["Date"], df["Visitor"], marker="o")
# 5. 设置标题和坐标轴
plt.title("Visitor Trend")
plt.xlabel("Date")
plt.ylabel("Visitor")
# 6. 调整日期标签
plt.xticks(rotation=45)
# 7. 自动调整布局并显示
plt.tight_layout()
plt.show()
# 8. 绘制温度与游客人数散点图
# 删除温度或游客人数缺失的行（仅用于绘图）
plot_df = df.dropna(subset=["Temperature", "Visitor"])
# 创建新的图表
plt.figure()
# 绘制散点图
plt.scatter(
    plot_df["Temperature"],
    plot_df["Visitor"]
)
# 设置标题和坐标轴
plt.title("Temperature vs Visitor")
plt.xlabel("Temperature")
plt.ylabel("Visitor")
# 显示图表
plt.show()
# 计算温度与游客人数的相关系数
correlation = plot_df["Temperature"].corr(
    plot_df["Visitor"]
)
print("温度与游客人数的相关系数：", correlation)
# 计算相关性矩阵
correlation_matrix = plot_df[
    ["Temperature", "Visitor"]
].corr()
# 输出相关性矩阵
print("相关性矩阵：")
print(correlation_matrix)
# 绘制相关性热力图
plt.figure(figsize=(6, 4))
plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
# 设置坐标轴
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)
# 添加相关系数数字
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(
            j, i,
            round(correlation_matrix.iloc[i, j], 2),
            ha="center",
            va="center"
        )
plt.colorbar()
plt.title("Correlation Heatmap")
plt.tight_layout()
# 保存热力图
plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
# 显示热力图
plt.show()