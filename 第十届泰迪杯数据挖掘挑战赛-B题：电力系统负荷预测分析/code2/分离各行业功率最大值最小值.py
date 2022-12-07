import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
#1商业
#2大工业用电
#3普通工业
#4非普工业
data = pd.read_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最大值.csv",encoding = 'gbk')
data1_max = data[['Time', '商业']]
data1_max = data1_max.rename(columns={'商业':'data'})
data1_max.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\商业有功功率最大值.csv",encoding = 'gbk')
#print(data1_max)

data2_max = data[['Time', '大工业用电']]
data2_max = data2_max.rename(columns={'大工业用电':'data'})
data2_max.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\大工业用电有功功率最大值.csv",encoding = 'gbk')

data3_max = data[['Time', '普通工业']]
data3_max = data3_max.rename(columns={'普通工业':'data'})
data3_max.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\普通工业有功功率最大值.csv",encoding = 'gbk')

data4_max = data[['Time', '非普工业']]
data4_max = data4_max.rename(columns={'非普工业':'data'})
data4_max.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\非普工业有功功率最大值.csv",encoding = 'gbk')



data = pd.read_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最小值.csv",encoding = 'gbk')
data1_min = data[['Time', '商业']]
data1_min = data1_min.rename(columns={'商业':'data'})
data1_min.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\商业有功功率最小值.csv",encoding = 'gbk')
#print(data1_max)

data2_min = data[['Time', '大工业用电']]
data2_min = data2_min.rename(columns={'大工业用电':'data'})
data2_min.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\大工业用电有功功率最小值.csv",encoding = 'gbk')

data3_min = data[['Time', '普通工业']]
data3_min = data3_min.rename(columns={'普通工业':'data'})
data3_min.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\普通工业有功功率最小值.csv",encoding = 'gbk')

data4_min = data[['Time', '非普工业']]
data4_min = data4_min.rename(columns={'非普工业':'data'})
data4_min.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\非普工业有功功率最小值.csv",encoding = 'gbk')