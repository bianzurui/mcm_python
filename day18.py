import matplotlib.pyplot as plt
def calculate_mae(y_test,y_pred):
    errors=[]
    for i in range(len(y_test)):
        error=abs(y_test[i]-y_pred[i])
        errors.append(error)
    mae=sum(errors)/len(errors)
    return mae
y_test=[100,200,300,400]
y_pred=[120,210,290,380]
baseline_pred=[100,200,300,300]
regression_pred=[120,210,290,380]
baseline_mae=calculate_mae(y_test,baseline_pred)
regression_mae=calculate_mae(y_test,regression_pred)
print("Baseline MAE:",baseline_mae)
print("回归模型 MAE:",regression_mae)
models=["Baseline","Regression"]
mae_values=[baseline_mae,regression_mae]
plt.bar(models,mae_values)
plt.bar_label(plt.gca().containers[0])
plt.ylabel("MAE")
plt.title("Model MAE Comparison")
plt.savefig("day18_mae_comparison.png",dpi=300,bbox_inches="tight")
plt.show()