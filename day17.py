import matplotlib.pyplot as plt
models=["Baseline","Regression"]
mae_values=[100,15]
plt.bar(models,mae_values)
plt.bar_label(plt.gca().containers[0])
plt.xlabel("Model Type")
plt.ylabel("Mean Absolute Error(MAE)")
plt.title("Comparision Of Model Prediction Errors")
plt.savefig("day17_mae_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

