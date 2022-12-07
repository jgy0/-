import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import optimizers
import time

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from chart_studio import plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
def creat_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)


dataframe = pd.read_csv('C:\\Users\Lenovo\Desktop\\有功功率最大值.csv',
                        header=0, parse_dates=[0],
                        index_col=0, usecols=[1, 2], squeeze=True)

dataframe.head(10)
import pandas as pd
df = pd.read_csv('C:\\Users\Lenovo\Desktop\附件2-行业日负荷数据.csv')


# In[37]:


metrics_df=pd.pivot_table(df,values='有功功率最大值（kw）',index='数据时间',columns='行业类型')
metrics_df.head()


# In[39]:


metrics_df.reset_index(inplace=True)
metrics_df.fillna(0,inplace=True)
metrics_df.head()
metrics_df = metrics_df.rename(columns={'数据时间':'Time'})
df = metrics_df
df['Time']= pd.to_datetime(df['Time'])
df.set_index('Time',inplace=True)
col = df.columns[0]
df.plot()


# In[44]:


col = df.columns[1]


# In[45]:

def plot_anomaly_window(ts, anomaly_pred=None, file_name='file', window='1h'):
    fig = go.Figure()
    yhat = go.Scatter(
        x=ts.index,
        y=ts,
        mode='lines', name=ts.name)
    fig.add_trace(yhat)
    if anomaly_pred is not None:
        for i in anomaly_pred.index:
            fig.add_vrect(x0=i - pd.Timedelta(window), x1=i, line_width=0, fillcolor="red", opacity=0.2)
    fig.show()

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3- Q1
c = 2
min_t = Q1 - c*IQR
max_t = Q3 + c*IQR
df[col+'threshold_alarm'] = (df[col].clip(lower = min_t,upper=max_t) != df[col])
plot_anomaly_window(df[col],anomaly_pred = df[df[col+'threshold_alarm']==True][col+'threshold_alarm'],window='1h',file_name = 'file')


# # Can we detect the noise?

# In[46]:


window = 5
df[col+'ma'] = df[col].rolling(window=window,closed='left').mean()
kpi_col = col+'ma'+'diff'
df[kpi_col] = (df[col]-df[col+'ma']).fillna(0)


Q1 = df[kpi_col].quantile(0.25)
Q3 = df[kpi_col].quantile(0.75)
IQR = Q3- Q1
c = 2
min_t = Q1 - c*IQR
max_t = Q3 + c*IQR
df[kpi_col+'threshold_alarm'] = (df[kpi_col].clip(lower = min_t,upper=max_t) != df[kpi_col])
plot_anomaly_window(df[col],anomaly_pred = df[df[kpi_col+'threshold_alarm']==True][kpi_col+'threshold_alarm'],file_name = 'file',window=f'{window}h')


# ## can we detect the condition/ level change?

# In[47]:


window = 10
df[col+'ma'] = df[col].rolling(window=window,closed='left').median()
df[col+'ma_shift'] = df[col+'ma'].shift(periods=window)
kpi_col = col+'ma'+'shift'+'diff'
df[kpi_col] = (df[col+'ma']-df[col+'ma_shift']).fillna(0)


Q1 = df[kpi_col].quantile(0.25)
Q3 = df[kpi_col].quantile(0.75)
IQR = Q3- Q1
c = 2
min_t = Q1 - c*IQR
max_t = Q3 + c*IQR
df[kpi_col+'threshold_alarm'] = (df[kpi_col].clip(lower = min_t,upper=max_t) != df[kpi_col])
plot_anomaly_window(df[col],anomaly_pred = df[df[kpi_col+'threshold_alarm']==True][kpi_col+'threshold_alarm'],file_name = 'file',window=f'{2*window}h')
