temperatures=[18,20,21,25,30]
count=0
total=0
for temperature in temperatures:
    if temperature>=20:
        count=count+1
        total=total+temperature
average=total/count
print("数量:",count)
print("总数：",total)
print("平均值：",average)
        

    