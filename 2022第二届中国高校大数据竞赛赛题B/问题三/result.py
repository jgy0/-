import pandas as pd
import numpy as np

df_2=pd.read_csv("../../code/submit.csv",header=None,encoding='utf-8')   # 任务二模型的预测结果
df_3=pd.read_csv("./submit3.csv",header=None,encoding='utf-8')           # 识别有无水印模型预测结果


# print(df_2.head(3).append(df_2.tail(3)))
# print('-'*8)
# print(df_3.head(3).append(df_3.tail(3)))
image={'图像编号':list(df_2.iloc[:,0])}
data=pd.DataFrame(image)
# print(data)
data['嵌入的信息']='无'
for i in range(len(data)):
    if df_3.iloc[i,1]==0:
        num=df_2.iloc[i,1]
        if num>=36 and num<=61:
            num=num+61
            data.iloc[i,1]=chr(num)
        elif num>=10 and num <=35:
            num=num+55
            data.iloc[i, 1] = chr(num)
        elif num>=0 and num <=9:
            num=num+48
            data.iloc[i, 1] = chr(num)

data.to_csv('./result.csv',index=None,encoding='utf_8_sig')





print(data)
data.info()
