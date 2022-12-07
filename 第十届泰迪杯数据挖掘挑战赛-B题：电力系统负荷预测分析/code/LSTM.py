# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

n_in = 960 #历史数量
n_out = 96 #预测数量
n_features = 1
# # n_test = 1
# n_val = 96
n_epochs = 3
path=r'D:\比赛\B题\train.csv'




df = pd.read_csv(path)

# df_1 = df.set_index('数据时间').asfreq('15T')
# df_1['总有功功率（kw）'] = df_1['总有功功率（kw）'].interpolate()

#
# df['数据时间'] = df['数据时间'].apply(lambda x: x.replace('/', ' ').replace(':', ' ').replace(' ', ''))
# df['数据时间'] = df['数据时间'].astype('int64')
df=df[['总有功功率（kw）']]
data=df.copy()
len(data)
kw = data['总有功功率（kw）'].values
kw.shape        #   (128156,)
kw = kw.reshape(len(kw), 1)    # (128156, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
kw = scaler.fit_transform(kw)
kw = kw.reshape(len(kw),)    # (128156,)
data['总有功功率（kw）'] = kw

train=data.copy()
# train = train.drop(["数据时间"], axis=1)
X_train, Y_train = [], []
for i in range(train.shape[0]-n_in-n_out+1):
    X_train.append(np.array(train.iloc[i:i+n_in]))
    Y_train.append(np.array(train.iloc[i+n_in:i+n_in+n_out]["总有功功率（kw）"]))

x=np.array(X_train)
y=np.array(Y_train)
print(len(X_train[0]),len(Y_train[0]),end=' ')
print(len(x[0]),len(y[0]),end=' ')
len(x[-1])
y.shape  # (127101, 96)
"""
划分训练集
"""
x_train = x[:-1]
x_train.shape   # (127488, 960, 1)
x_test = x[-1:]
x_test.shape    # (1, 960, 1)
y_train = y[:-1]
y_train.shape   # (127488, 96)
y_test = y[-1:]
y_test.shape   #  (1, 96)
"""
搭建模型
"""
model = Sequential()
model.add(LSTM(12, activation='relu', input_shape=(n_in, n_features)))
model.add(Dropout(0.3))
model.add(Dense(n_out))
model.compile(optimizer='adam', loss='mae')
model.fit(x_train, y_train, epochs=n_epochs, batch_size=1, verbose=1)
# m = model.evaluate(x_test, y_test)
saver=tf.train.Saver()
saver.save(model,'LSTM.ckpt')
# 预测
predict=model.predict(x_test)

validation = scaler.inverse_transform(predict)[0]
actual = scaler.inverse_transform(y_test)[0]
predict = validation
actual = actual
print(predict)
print(actual)
x = [x for x in range(96)]
fig, ax = plt.subplots(figsize=(15,5),dpi = 300)
ax.plot(x, predict, linewidth=2.0,label = "predict")
ax.plot(x, actual, linewidth=2.0,label = "actual")
ax.legend(loc=2)
plt.ylim((0, 400000))
plt.grid(linestyle='-.')
plt.show()

error = 0
summery = 0
for i in range(24):
    error += abs(predict[i] - actual[i])
    summery += actual[i]
acc = 1 - error/summery
print(acc)






