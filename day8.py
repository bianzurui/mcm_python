
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. 读取数据
df = pd.read_csv("visitor_multivariate_regression_data.csv")

# 2. 选择影响因素和预测目标
X = df[["Temperature", "Rain", "Holiday"]]
y = df["Visitor"]

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 预测测试集
y_pred = model.predict(X_test)

# 6. 计算残差
residuals = y_test - y_pred

print("实际游客数：")
print(y_test.to_list())

print("预测游客数：")
print(y_pred)

print("残差：")
print(residuals.to_list())

# 7. 绘制残差图
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Visitors")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()