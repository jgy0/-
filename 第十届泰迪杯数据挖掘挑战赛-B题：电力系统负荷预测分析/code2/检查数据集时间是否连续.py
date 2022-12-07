import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time, datetime
from datetime import datetime, date, timedelta
import pandas as pd
data = pd.read_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\附件2-行业日负荷数据副本.csv",encoding = 'gbk')
x = data[data.行业类型.apply(lambda x: x == '大工业用电')].数据时间
#y = data[data.行业类型 == "大工业用电"].有功功率最大值（kw）
y = data[data.行业类型.apply(lambda x: x == '大工业用电')].有功功率最大值
print(y)
plt.scatter(x, y)
plt.plot(x, y)
plt.show()
insert_x = []
lxs = len(x)
#if lxs < 10:
    # 判断是否是连续日期，并对不连续的日期进行时间插值
for i in range(len(x)):
        if i + 1 == len(x):
            break
        t1 = int(time.mktime(time.strptime(x[i], "%Y/%m/%d")))
        t2 = int(time.mktime(time.strptime(x[i + 1], "%Y/%m/%d")))
        differ = (datetime.fromtimestamp(t2) - datetime.fromtimestamp(t1)).days
        # print("相差",differ,"天")
        while differ != 1:
            differ -= 1
            tmp = (datetime.fromtimestamp(t2) + timedelta(days=-differ)).strftime("%Y/%m/%d")
            insert_x.append(tmp)

print("缺失日期为：",insert_x)

