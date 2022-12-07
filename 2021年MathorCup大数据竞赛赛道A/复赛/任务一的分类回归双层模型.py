import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')
"""
评估数据集的构造
"""
# df_5 = pd.read_csv('./data/附件5：门店交易验证数据.txt', sep='\t', parse_dates=[1], header=None, encoding='utf-8')   # 附件五
# df_2 = pd.read_csv('../data/附件2：估价验证数据.txt', sep='\t', parse_dates=[1, 11, 12], header=None, encoding='utf-8')   # 附件二的信息

df_5 = pd.read_csv(r'D:\应用\pycharm\PyCharm 2021.1.3\source\mathorcup\复赛\data\附件5：门店交易验证数据.txt', sep='\t', parse_dates=[1], header=None, encoding='utf-8')   # 附件五
df_2 = pd.read_csv(r'D:\应用\pycharm\PyCharm 2021.1.3\source\mathorcup\data\附件2：估价验证数据.txt', sep='\t', parse_dates=[1, 11, 12], header=None, encoding='utf-8')   # 附件二的信息

columns = ['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode',
           'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear',
           'displacement', 'gearbox', 'oiltype', 'newprice']
for i in range(1,16):
    columns.append('anonymousFeature'+str(i))
# 重命名数据列名
df_2.columns = columns
# df_5.to_csv(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\复赛\附件五.csv', encoding='utf-8')
df_5.rename(columns={0:'carid',1:'pushDate',2:'pushPrice',3:'updatePriceTimeJson'} ,inplace=True)
# df_5.to_csv(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\复赛\附件五1.csv', encoding='utf-8')
print(df_2)
df_test = pd.merge(df_5, df_2, on='carid', how='left')
# df_test.to_csv(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\复赛\任务一test.csv')
"""
训练集的构造
"""
# df_1 = pd.read_csv('../data/附件1：估价训练数据.txt', sep='\t', parse_dates=[1, 11, 12], encoding='utf-8', header=None)  # 附件一
# df_4 = pd.read_csv('../data/附件4：门店交易训练数据.txt', sep='\t', parse_dates=[1, 4, 5], encoding='utf-8', header=None)   # 附件四

df_1 = pd.read_csv(r'D:\应用\pycharm\PyCharm 2021.1.3\source\mathorcup\data\附件1：估价训练数据.txt', sep='\t', parse_dates=[1, 11, 12], encoding='utf-8', header=None)  # 附件一
df_4 = pd.read_csv(r'D:\应用\pycharm\PyCharm 2021.1.3\source\mathorcup\data\附件4：门店交易训练数据.txt', sep='\t', parse_dates=[1, 4, 5], encoding='utf-8', header=None)   # 附件四


columns.append('price')
df_1.columns=columns
df_4.rename(columns={0:'carid',1:'pushDate',2:'pushPrice',3:'updatePriceTimeJson',4:'pullDate',5:'withdrawDate'},inplace=True)
df_train = pd.merge(df_4, df_1, how='left', on='carid')

# print(df_train.columns)
# print(df_test.columns)

# 调整trian的数据特征，保持与test一致
df_train['cycle'] = df_train['withdrawDate']-df_train['pushDate']
df_train=df_train.drop(['price', 'pullDate', 'withdrawDate'], axis=1)
df_train['cycle']=df_train['cycle'].astype('timedelta64[D]')
# df_train.to_csv(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\复赛\任务一train.csv')

"""
数据处理
"""
df_train.dropna(how='any',subset=['carCode','gearbox'],inplace=True)  # 训练集中有缺失值，缺失值较少 ：6个，1个

def fill_(data):
    n=['country','maketype','modelyear']
    for i in n:
        x=int(data[i].mode())
        data[i].fillna(x,inplace=True)
    return data
df_train=fill_(df_train)
df_test=fill_(df_test)

df_train.drop(['anonymousFeature4','anonymousFeature7','anonymousFeature15'],axis=1,inplace=True)
df_test.drop(['anonymousFeature4','anonymousFeature7','anonymousFeature15'],axis=1,inplace=True)
"""
异常值检验，
"""



"""
特征工程
"""
def deal_ano_feature(df_train):
    df_train['anonymousFeature11'].fillna('1+2',inplace=True)
    def deal_11(x):
        return sum([float(x) for x in re.findall("\d",x)])
    df_train['anonymousFeature11']=df_train['anonymousFeature11'].map(deal_11)

    # 处理匿名特征12
    def deal_12(x):
        li=[float(x) for x in re.findall("\d+",x)]
        return li[0]*li[1]*li[2]
    df_train['anonymousFeature12_length']=df_train['anonymousFeature12'].apply(lambda x:int(x.split('*')[0]))
    df_train['anonymousFeature12_width']=df_train['anonymousFeature12'].apply(lambda x:int(x.split('*')[1]))
    df_train['anonymousFeature12_height']=df_train['anonymousFeature12'].apply(lambda x:int(x.split('*')[2]))
    df_train['anonymousFeature12']=df_train['anonymousFeature12'].map(deal_12)
    # 处理匿名特征13
    df_train['anonymousFeature13'].fillna(int(df_train['anonymousFeature13'].mode()),inplace=True)
    def deal_13(x):
        return x[:4], x[4:6]
    df_train['anonymousFeature13']=df_train['anonymousFeature13'].astype('string')
    df_train['anonymousFeature13_year']=df_train['anonymousFeature13'].map(deal_13)    #  (2017, 09)
    df_train['anonymousFeature13_month']=df_train['anonymousFeature13_year'].apply(lambda x: int(x[1]))
    df_train['anonymousFeature13_year']=df_train['anonymousFeature13_year'].apply(lambda x: int(x[0]))
    df_train['anonymousFeature13']=df_train['anonymousFeature13'].astype('float')

    df_train.isnull().sum()

    # df_train.dtypes

    def deal(data=None):
        li=[]
        for i in data.index:
            count=len(re.findall('\d+-\d+-\d+',str(data.loc[i,'updatePriceTimeJson'])))
            li.append(count)
        return li

    df_train['decreasing_count']=deal(df_train)

    # df_train['decreasing_count'].unique()
    # df_train['decreasing_count'].value_counts()

    df_train.isnull().sum()
    """
    对剩余的缺失值进行填充
    """
    # 匿名变量1 , 8 ,9 ,10先用众数填充；   或者删除有缺失值的行
    def fill_v(data):
        n=['anonymousFeature1','anonymousFeature8','anonymousFeature9','anonymousFeature10']
        for i in n:
            x=int(data[i].mode())
            data[i].fillna(x,inplace=True)
        return data

    df_train=fill_v(df_train)
    df_train['pushDate_year'] = df_train['pushDate'].dt.year
    df_train['pushDate_month'] = df_train['pushDate'].dt.month
    df_train['pushDate_day'] = df_train['pushDate'].dt.day
    return df_train

df_train=deal_ano_feature(df_train)
df_test=deal_ano_feature(df_test)
# 将cycle调整到最后一列
cycle=df_train['cycle']
df_train=df_train.drop('cycle',axis=1)
df_train['cycle']=cycle
"""
以上思路是清晰的
"""
# 分类
df_train_class=df_train.copy()    # 让文件指向不同的值
df_test_class=df_test.copy()
"""
开始分开处理
"""
# df_train_class is df_train
df_train_class['label']=np.where(df_train_class['cycle'].isnull(),-1,1)
# df_train_class.dtypes
df_train_class=df_train_class.select_dtypes(include=['float64', 'int64', 'int32'])
df_train_class=df_train_class.drop(['cycle','carid'],axis=1)

df_test_class=df_test_class.select_dtypes(include=['float64', 'int64', 'int32'])
df_test_class=df_test_class.drop('carid',axis=1)

# df_train_class.dtypes
# df_test_class.dtypes
# df_train_class.isnull().sum()
# df_test_class.isnull().sum()
"""
X,y的数据已经满足机器学习的要求了
"""

pearson=df_train_class.corr()
index = pearson['label'][:-1].abs() > 0.03
X=df_train_class.iloc[:,:-1]
X_submit=X.loc[:,index]
feature=X_submit.columns

df_test_class=df_test_class[feature]
# df_test_class.columns
"""
分类模型,判断是否能卖出
"""

X_train,X_test,y_train,y_test=train_test_split(X_submit,df_train_class['label'],test_size=0.2)

model=RandomForestClassifier(n_estimators=300,random_state=33,n_jobs=-1)
model.fit(X_train,y_train)
model_path='./models'
joblib.dump(model,f"{model_path}/model.joblib")
y_pred=model.predict(X_test)
# sum(y_pred==y_test)/len(y_pred)
res=classification_report(y_test,y_pred)
print(res)

"""
以上思路清楚
"""



y_=model.predict(df_test_class)
df_test['label']=y_
# data=pd.DataFrame(y_)
# data[0].unique()
# data[0].value_counts()
df_test['label'].value_counts()
"""
 1    1918
-1      82
Name: label, dtype: int64

"""

"""
对分类模型判断可以卖出的数据，进行回归预测交易周期
"""
df_train_reg_ori=df_train
df_test_reg_ori=df_test

# 预测卖不出去的结果保存
res_test_1=df_test[df_test['label']==-1].copy()
res_test_1.rename(columns={'label':'cycle'},inplace=True)

"""
可以卖出的回归预测即可
"""
# 删除cycle为空的，
df_train_reg_ori=df_train_reg_ori.dropna()
# df_train_reg_ori.isnull().sum()
# df_train_reg_ori.dtypes

df_train_reg=df_train_reg_ori.select_dtypes(include=['float64','int64'])
pearson_reg=df_train_reg.corr()
index_reg=pearson_reg['cycle'][:-1].abs() > 0.04
X_=df_train_reg.iloc[:,:-1]
# X_.columns
X_submit_reg=X_.loc[:,index_reg]

X_submit_reg.columns
X_train_reg,X_test_reg,y_train_reg,y_test_reg=train_test_split(X_submit_reg,df_train_reg['cycle'])

model_reg=RandomForestRegressor(n_estimators=500,random_state=33,n_jobs=-1)
model_reg.fit(X_train_reg,y_train_reg)
joblib.dump(model_reg,f"{model_path}/model_reg.joblib")
y_reg=model_reg.predict(X_test_reg)
res=mean_absolute_error(y_test_reg,y_reg)
print(res)

data_0=df_test[df_test['label']==1].copy()   # 取出经分类模型判断cycle不为空的test数据
data=data_0.select_dtypes(include=['float64','int64'])
# data.columns
data=data.loc[:,index_reg]

y_reg_=model_reg.predict(data)
data_0['cycle']=y_reg_
# data_0['label'].unique()
data_0=data_0.drop('label',axis=1)

finall_res=pd.concat([data_0,res_test_1])   # 把分类的结果和回归的结果文件拼接到一起

finall_res=finall_res[['carid','cycle']]

finall_res.to_csv(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\复赛\附件6：门店交易模型结果.txt',sep='\t',index=None,header=None)






# # 取出经过分类模型预测的测试集中可以卖出的数据
# df_test_reg=df_test_reg_ori[df_test['label']==1]
# len(df_test_reg)
# df_test_reg=df_test_reg_ori.drop('label',axis=1)
#
# df_test_reg_1=df_test_reg.drop('carid', axis=1)     # 去除id
# df_test_reg=df_test_reg_1.copy()
# # df_train_reg.columns
# # df_test_reg.columns
# df_train_reg=df_train_reg.select_dtypes(include=['float64', 'int64', 'int32'])
# df_test_reg=df_test_reg.select_dtypes(include=['float64', 'int64', 'int32'])
#
# len(df_train_reg)
#
# data_feature=SelectKBest(chi2,k=10).fit_transform(df_train_reg.iloc[:,:-1],df_train_reg.iloc[:,-1])
# data_feature_test=SelectKBest(chi2,k=10).fit_transform(df_test_reg.iloc[:,:-1],df_test_reg.iloc[:,-1])
#
# X_train_reg,X_test_reg,y_train_reg,y_test_reg=train_test_split(data_feature,df_train_reg.iloc[:,-1])
#
# model_reg=RandomForestRegressor(n_estimators=500,random_state=33,n_jobs=-1)
# model_reg.fit(X_train_reg,y_train_reg)
# y_reg=model_reg.predict(data_feature_test)
#
# len(y_reg)
# len(y_test_reg)
# mean_absolute_error(y_test_reg,y_reg)
#
# df_test_reg_ori['cycle']=y_reg
#
# res_data=pd.concat([res_test_1,df_test_reg_ori])
# print(res_data)
# res_data=res_data[['carid','cycle']]
#
# res_data.to_csv('./data/附件6：门店交易模型结果.txt',sep='\t',index=None,header=None)
