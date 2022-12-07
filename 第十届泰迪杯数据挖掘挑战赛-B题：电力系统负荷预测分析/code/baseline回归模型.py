# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score,mean_absolute_error



# df_1=pd.read_csv('../data/附件1-区域15分钟负荷数据.csv',encoding='utf-8',parse_dates=['数据时间'])
df_1=pd.read_csv(r'D:\应用\pycharm\PyCharm 2021.1.3\source\TaiDi\data\附件1-区域15分钟负荷数据.csv'
                 ,encoding='utf-8',parse_dates=['数据时间'])
# df_1.info()
# df_2=pd.read_csv('../data/附件3-气象数据.csv',encoding='utf-8')
df_2=pd.read_csv(r'D:\应用\pycharm\PyCharm 2021.1.3\source\TaiDi\data\附件3-气象数据.csv',encoding='utf-8')
"""
数据处理
"""
# 去除重复值
print("附件一原始的数据量:{}".format(len(df_1)))
df_1=df_1.drop_duplicates()
print("附件一去重后的数据量:{}".format(len(df_1)))
print("附件三原始的数据量:{}".format(len(df_2)))
df_2=df_2.drop_duplicates()
print("附件三去重后的数据量:{}".format(len(df_2)))
## 填补缺失值
df_1 = df_1.set_index('数据时间').asfreq('15T')
df_1.isnull().sum()
df_1['总有功功率（kw）'] = df_1['总有功功率（kw）'].interpolate()
df_1.isnull().sum()

plt.plot(df_1.index[:3000],df_1['总有功功率（kw）'][:3000])
plt.show()
df_1=df_1.reset_index()

df_2['日期']=df_2['日期'].apply(lambda x: x.replace('年','/').replace('月','/').replace('日',''))
df_2['日期']=pd.to_datetime(df_2['日期'])

# df_2=df_2.drop(['Unnamed: 6'],axis=1)
# print(df_2)
df_2['最高温度']=df_2['最高温度'].apply(lambda x: x.replace('℃',''))
df_2['最低温度']=df_2['最低温度'].apply(lambda x: x.replace('℃',''))
df_2['最高温度']=df_2['最高温度'].astype(float)
df_2['最低温度']=df_2['最低温度'].astype(float)

# 处理天气情况
num=df_2['天气状况'].str.split('/',expand=True)
# num2=df_2.join(df_2['天气状况'].str.split('/',expand=True))
"""
加上join  直接合并了
"""
df_2['天气1'],df_2['天气2']=num[0],num[1]

df_2['白天风力风向'].unique()
df_2['夜晚风力风向'].unique()
df_2['天气1'].unique()
df_2['天气2'].unique()


n=['天气1','天气2','白天风力风向','夜晚风力风向']
for i in n:
    df_2[i] = df_2[i].map(df_2[i].value_counts())
# df_2.info()
del df_2['天气状况']


df_1['left_on']=df_1['数据时间'].dt.date
df_2['right_on']=df_2['日期'].dt.date
print(df_1)
print(df_2)
data=pd.merge(df_1,df_2,how='left',left_on=['left_on'],right_on='right_on')
print(data)
data=data.drop(['left_on','right_on','日期'],axis=1)
print(data)
# data.to_csv(r'D:\比赛\B题\data.csv',index=None)
train_data=data.copy()

train_data['年']=train_data['数据时间'].dt.year
train_data['月']=train_data['数据时间'].dt.month
train_data['日']=train_data['数据时间'].dt.day
train_data['小时']=train_data['数据时间'].dt.hour
train_data['一年中的第几周']=train_data['数据时间'].dt.isocalendar().week

train_data['月末']=train_data['数据时间'].dt.is_month_end
train_data['月初']=train_data['数据时间'].dt.is_month_start
train_data['月末']=train_data['月末'].astype(int)
train_data['月初']=train_data['月初'].astype(int)
train_data['季末']=train_data['数据时间'].dt.is_quarter_end
train_data['季初']=train_data['数据时间'].dt.is_quarter_start
train_data['季末']=train_data['季末'].astype(int)
train_data['季初']=train_data['季初'].astype(int)
train_data['周末']=train_data['数据时间'].dt.dayofweek
# print(train_data['周末'][10000])
train_data['周末']=train_data['周末'].apply(lambda x: 1 if x>4 else 0)

print(train_data.dtypes)
train_data['一年中的第几周']=train_data['一年中的第几周'].astype(int)
# train_data.to_csv(r'D:\比赛\B题\train.csv',index=None)

y=train_data['总有功功率（kw）']
del train_data['总有功功率（kw）']
train_data.insert(len(train_data.columns),'总有功功率（kw）',y)
del train_data['数据时间']
"""
特征选择
"""
# pearson=train_data.corr()
# index=pearson['总有功功率（kw）'].abs() > 0.1
# data_finall=train_data.loc[:,index]
# data_finall.columns
#
# del data_finall['总有功功率（kw）']
train_data.dtypes
X=train_data.select_dtypes(include=['int32','int64','float64'])
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor

data_finall=SelectFromModel(GradientBoostingRegressor()).fit_transform(X.iloc[:,:-1],y)

"""
划分训练集和测试集 
"""

X_train=data_finall[:100000]
y_train=y[:100000]
y_train=np.log(y_train)

X_test=data_finall[100000:]
y_test=y[100000:]


model_lgb = lgb.LGBMRegressor(
                learning_rate=0.01,
                max_depth=-1,
                n_estimators=1000,
                boosting_type='gbdt',
                random_state=2021,
                objective='regression',
                num_leaves=32,
                verbose=-1)
lgb_model = model_lgb.fit(X_train,y_train)
y_pred = np.expm1(lgb_model.predict(X_test))
y_test

print("R2_socre:{}".format(r2_score(y_test,y_pred)))
# 0.20503225628709132
print("MAE:{}".format(mean_absolute_error(y_test, y_pred)))
# 22291.535005275095
"""
使用基于树模型的特征选择
R2_socre:0.35966756065094785
MAE:21552.17419717049
"""
"""
取300绘图
"""

y_p=y_pred[:300]
y_t=y_test[:300]
plt.plot(range(len(y_p)),y_p,'r-.')
plt.plot(range(len(y_t)),y_t,'g-*')
plt.legend(['y_pred','y_true'])
plt.show()
mean_squared_error(y_test,y_pred)


# def test(n:int,age:'int > 0'=20)-> int:           # 注解表达式
#     return n,age
#
#
# test(1,2)
# a,b=test('www')
# print(test.__annotations__)
#
#
# l=[1,2,3]
# print(*l)
#
# nn=[word for word in input()[:].split()]

