mae_a=25
mae_b=18
improvement=mae_a-mae_b
improvement_rate=improvement/mae_a*100
print("模型A的MAE:",mae_a)
print("模型B的MAE:",mae_b)
print("误差下降量：",improvement)
print("误差相对下降比例：",improvement_rate)
models=["模型A","模型B","模型C"]
mae_values=[25,18,22]
best_index=mae_values.index(min(mae_values))
print("最好的模型是：",models[best_index])
print("对应的验证集MAE:",mae_values[best_index])
import matplotlib.pyplot as plt
models = ["模型 A", "模型 B", "模型 C"]
mae_values = [16, 21, 19]
bars=plt.bar(models,mae_values)
plt.bar_label(bars)
plt.xlabel("模型")
plt.ylabel("验证集MAE")
plt.title("模型表现比较")
plt.show()