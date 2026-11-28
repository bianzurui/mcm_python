import pandas as pd
import matplotlib.pyplot as plt
# 1. 读取真实数据
df = pd.read_csv("MCM_2026_C_Data.csv")
# 2. 提取游客人数
visitor = df["Visitor"].dropna()
# 3. 计算第一、第三四分位数
q1 = visitor.quantile(0.25)
q3 = visitor.quantile(0.75)
print("第一四分位数 Q1：", q1)
print("第三四分位数 Q3：", q3)
# 4. 计算四分位距
iqr = q3 - q1
print("四分位距 IQR：", iqr)
# 5. 计算异常值上下界
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print("异常值下界：", lower_bound)
print("异常值上界：", upper_bound)
# 6. 判断哪些数据低于下界
below = visitor < lower_bound
# 7. 判断哪些数据高于上界
above = visitor > upper_bound
print("低于下界的数据：")
print(below)
print("高于上界的数据：")
print(above)
# 9. 人为添加一个800，进行异常值测试
test_visitor = pd.concat(
    [visitor, pd.Series([800])],
    ignore_index=True
)
print("加入800后的测试数据：")
print(test_visitor)
# 10. 使用IQR方法识别异常值
test_outliers = test_visitor[
    (test_visitor < lower_bound) |
    (test_visitor > upper_bound)
]
print("识别出的异常值：")
print(test_outliers)
# 12. 比较原始数据和测试数据
plt.figure(figsize=(7, 5))
plt.boxplot(
    [visitor, test_visitor],
    tick_labels=["Original", "With Outlier"]
)
plt.title("Visitor Comparison")
plt.ylabel("Visitor")
plt.show()
# 13. 计算平均值
mean_value = visitor.mean()
# 14. 计算标准差
std_value = visitor.std()
print("游客人数平均值：", mean_value)
print("游客人数标准差：", std_value)
# 15. 计算每个数据的Z-score
z_scores = (test_visitor - mean_value) / std_value
print("每个数据的Z-score：")
print(z_scores)
# 16. 筛选Z-score绝对值大于3的数据
z_outliers = test_visitor[abs(z_scores) > 3]
print("Z-score识别出的异常值：")
print(z_outliers)
# 17. 使用IQR方法识别异常值
iqr_outliers = test_visitor[
    (test_visitor < lower_bound) |
    (test_visitor > upper_bound)
]
# 18. 使用Z-score方法识别异常值
z_outliers = test_visitor[
    abs(z_scores) > 3
]
# 19. 比较两种方法
print("IQR识别结果：")
print(iqr_outliers)
print("Z-score识别结果：")
print(z_outliers)
# 20. 综合比较两种异常值识别方法

# IQR方法
iqr_outliers = test_visitor[
    (test_visitor < lower_bound) |
    (test_visitor > upper_bound)
]

# Z-score方法
z_outliers = test_visitor[
    abs(z_scores) > 3
]

# 输出识别结果
print("========== 异常值识别对比 ==========")

print("IQR识别结果：")
print(iqr_outliers)

print("Z-score识别结果：")
print(z_outliers)

# 统计识别数量
print("IQR识别数量：", len(iqr_outliers))
print("Z-score识别数量：", len(z_outliers))
# Day4最终实战：检查原始数据

# IQR方法
original_iqr_outliers = visitor[
    (visitor < lower_bound) |
    (visitor > upper_bound)
]

# Z-score方法
original_z_scores = (
    visitor - mean_value
) / std_value

original_z_outliers = visitor[
    abs(original_z_scores) > 3
]

print("===== 原始数据检测 =====")

print("IQR识别结果：")
print(original_iqr_outliers)

print("Z-score识别结果：")
print(original_z_outliers)