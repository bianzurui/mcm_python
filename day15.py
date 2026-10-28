y_test=[100,200,300,400]
y_train=[100,200,300,400,500]
train_mean=sum(y_train)/len(y_train)
y_pred=[train_mean]*len(y_test)
print("训练集平均值：",train_mean)
print("预测值：",y_pred)
errors=[]
for i in range(len(y_test)):
    error=abs(y_test[i]-y_pred[i])
    errors.append(error)
mae=sum(errors)/len(errors)
print("errors",errors)
print("MAE",mae)