y_test=[100,200,300,400]
y_pred=[120,210,290,380]
errors=[]
for i in range(len(y_test)):
    error=abs(y_test[i]-y_pred[i])
    errors.append(error)
mae=sum(errors)/len(errors)
print("每个样本的绝对误差：",errors)
max_error=max(errors)
print("最大绝对误差：",max_error)
for i in range(len(errors)):
    if errors[i]==max_error:
        print("最大误差是第",i+1,"天")

