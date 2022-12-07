import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import optimizers
import time

#--------------------------------------------#
#   得到时间序列+某一行业的一列数据
#--------------------------------------------#
dataframe = pd.read_csv('C:\\Users\Lenovo\Desktop\\有功功率最小值.csv',
                        header=0, parse_dates=[0],
                        index_col=0, usecols=[1, 2], squeeze=True)
dataset = dataframe.values
print(dataframe.head(10))

#--------------------------------------------#
#   画出每个行业功率最大（小）值随时间变化的折线图
#--------------------------------------------#
#————————————商业
dataframe = pd.read_csv('C:\\Users\Lenovo\Desktop\\有功功率最小值.csv')
dataframe['Time']=pd.to_datetime(dataframe['Time'])
series = dataframe.set_index(['Time'], drop=True)
plt.figure(figsize=(10, 6))
plt.title("商业")
series['商业'].plot()
plt.show()

#————————————大工业用电
dataframe = pd.read_csv('C:\\Users\Lenovo\Desktop\\有功功率最小值.csv')
dataframe['Time']=pd.to_datetime(dataframe['Time'])
series = dataframe.set_index(['Time'], drop=True)
plt.figure(figsize=(10, 6))
plt.title("大工业用电")
series['大工业用电'].plot()

plt.show()

#————————————普通工业
dataframe = pd.read_csv('C:\\Users\Lenovo\Desktop\\有功功率最小值.csv')
dataframe['Time']=pd.to_datetime(dataframe['Time'])
series = dataframe.set_index(['Time'], drop=True)
plt.figure(figsize=(10, 6))
plt.title("普通工业")
series['普通工业'].plot()
plt.show()

#————————————非普工业
dataframe = pd.read_csv('C:\\Users\Lenovo\Desktop\\有功功率最小值.csv')
dataframe['Time']=pd.to_datetime(dataframe['Time'])
series = dataframe.set_index(['Time'], drop=True)
plt.figure(figsize=(10, 6))
plt.title("非普工业")
series['非普工业'].plot()
plt.show()

from statsmodels.tsa.stattools import adfuller










