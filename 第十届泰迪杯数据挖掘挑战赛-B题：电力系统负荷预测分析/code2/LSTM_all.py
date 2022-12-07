import pandas as pd
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
plt.rcParams['font.sans-serif'] = ['SimHei']
for filename in os.listdir(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\各行业最大最小功率"):
                 print(filename)
                 path = "C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\各行业最大最小功率/%s"%(filename)
                 data = pd.read_csv(path,encoding = 'gbk')
                 print(data)
                 data['Time'] = pd.to_datetime(data['Time'])
                 print(data)
                 df = data
                 train_size = int(0.85 * len(df))
                 test_size = len(df) - train_size

                 df = df.fillna(0)
                 univariate_df = df[['Time', 'data']].copy()
                 univariate_df.columns = ['ds', 'y']

                 train = univariate_df.iloc[:train_size, :]

                 x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(
                     univariate_df.iloc[:train_size, 1])
                 #print(x_train)
                 x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(
                     univariate_df.iloc[train_size:, 1])

                 #print(len(train), len(x_valid))

                 from sklearn.preprocessing import MinMaxScaler

                 data = univariate_df.filter(['y'])
                 # Convert the dataframe to a numpy array
                 dataset = data.values

                 scaler = MinMaxScaler(feature_range=(-1, 0))
                 scaled_data = scaler.fit_transform(dataset)

                 scaled_data[:10]

                 scaler = MinMaxScaler(feature_range=(-1, 0))
                 scaled_data = scaler.fit_transform(dataset)

                 print(scaled_data)

                 # Defines the rolling window
                 look_back = 60
                 # Split into train and test sets
                 train, test = scaled_data[:train_size - look_back, :], scaled_data[train_size - look_back:, :]


                 def create_dataset(dataset, look_back=1):
                     X, Y = [], []
                     for i in range(look_back, len(dataset)):
                         a = dataset[i - look_back:i, 0]
                         X.append(a)
                         Y.append(dataset[i, 0])
                     return np.array(X), np.array(Y)


                 x_train, y_train = create_dataset(train, look_back)
                 x_test, y_test = create_dataset(test, look_back)

                 # reshape input to be [samples, time steps, features]
                 x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
                 x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

                 #print(len(x_train), len(x_test))
                 #print(y_train)

                 from tensorflow.keras.models import Sequential
                 from tensorflow.keras.layers import Dense, LSTM

                 # Build the LSTM model
                 model = Sequential()
                 model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                 model.add(LSTM(64, return_sequences=False))
                 model.add(Dense(25))
                 model.add(Dense(1))

                 # Compile the model
                 model.compile(optimizer='adam', loss='mean_squared_error')

                 # Train the model
                 model.fit(x_train, y_train, batch_size=15, epochs=30, validation_data=(x_test, y_test))
                 model.summary()

                 # Lets predict with the model
                 train_predict = model.predict(x_train)
                 test_predict = model.predict(x_test)

                 # invert predictions
                 train_predict = scaler.inverse_transform(train_predict)
                 y_train = scaler.inverse_transform([y_train])

                 test_predict = scaler.inverse_transform(test_predict)
                 y_test = scaler.inverse_transform([y_test])

                 # Get the root mean squared error (RMSE) and MAE
                 score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
                 score_mae = mean_absolute_error(y_test[0], test_predict[:, 0])
                 print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))
                 from sklearn.metrics import r2_score
                 print('R2-score:', r2_score(y_test[0], test_predict[:, 0]))
                 r2_score = r2_score(y_test[0], test_predict[:, 0])

                 x_train_ticks = univariate_df.head(train_size)['ds']
                 y_train = univariate_df.head(train_size)['y']
                 x_test_ticks = univariate_df.tail(test_size)['ds']

                 # Plot the forecast
                 f, ax = plt.subplots(1)
                 f.set_figheight(6)
                 f.set_figwidth(15)

                 sns.lineplot(x=x_train_ticks, y=y_train, ax=ax, label='Train Set')  # navajowhite
                 sns.lineplot(x=x_test_ticks, y=test_predict[:, 0], ax=ax, color='green',
                              label='Prediction')  # navajowhite
                 sns.lineplot(x=x_test_ticks, y=y_test[0], ax=ax, color='orange', label='Ground truth')  # navajowhite

                 ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f},R2-score: {r2_score:.2f}', fontsize=14)
                 ax.set_xlabel(xlabel='Date', fontsize=14)
                 ax.set_ylabel(ylabel=filename[:-4], fontsize=14)
                 path = "C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\各行业预测最大最小功率/%s" % (filename)
                 test_predict = pd.DataFrame(test_predict)
                 test_predict.to_csv(path, encoding='gbk')
                 plt.savefig("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\各行业预测最大最小功率/%s.png" % (filename[:-4]))
                 print("%s完成"%(filename))
                 #plt.show()
                 #.render(path="C:/Users/Lenovo/Desktop/上下行吞吐量图/%s.html" % (filename))