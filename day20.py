import matplotlib.pyplot as plt
days=["第一天","第二天","第三天","第四天","第五天"]
errors=[10,30,10,50,10]
bars=plt.bar(days,errors)
plt.bar_label(bars)
plt.xlabel("天数")
plt.ylabel("绝对误差")
plt.title("预测误差分析")
plt.show()
