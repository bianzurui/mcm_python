import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv("MCM_2026_C_Data.csv")
df_clean=df.dropna()
x=df_clean["Temperature"]
y=df_clean["Visitor"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
slope,intercept=np.polyfit(x_train,y_train,1)
y_pred=slope*x_test+intercept
mae=np.mean(abs(y_test-y_pred))
print("测试集 MAE:",mae)
rmse=np.sqrt(np.mean((y_test-y_pred)**2))
print("测试集 RMSE:",rmse)