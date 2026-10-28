y_test=[100,200,300,400]
baseline_pred=[300,300,300,300]
regression_pred=[120,210,290,380]
print("真实值：",y_test)
print("baseline预测:",baseline_pred)
print("回归模型预测：",regression_pred)
errors=[]
for i in range(len(y_test)):
    error=abs(y_test[i]-baseline_pred[i])
    errors.append(error)
baseline_mae=sum(errors)/len(errors)
print("baseline_mae:",baseline_mae)
errors=[]
for i in range(len(y_test)):
    error=abs(y_test[i]-regression_pred[i])
    errors.append(error)
regression_mae=sum(errors)/len(errors)
print("回归模型MAE：",regression_mae)
print("模型比较结果：")
if regression_mae<baseline_mae:
    print("回归模型的MAE更小，预测误差更低")
elif regression_mae>baseline_mae:
    print("baseline的MAE更小，预测误差更低")
else:
    print("两个模型的MAE相同")
