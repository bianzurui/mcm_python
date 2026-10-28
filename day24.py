import matplotlib.pyplot as plt
models=["模型A","模型B"]
mae_values=[1.70,18.35]
bars=plt.bar(models,mae_values)
plt.bar_label(bars)
plt.xlabel("模型")
plt.ylabel("平均MAE")
plt.title("模型误差比较")
plt.savefig("day24_model_comparision.png",dpi=300,bbox_inches="tight")
plt.show()