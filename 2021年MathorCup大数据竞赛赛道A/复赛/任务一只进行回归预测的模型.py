import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest
import xgboost as xgb
import lightgbm as lgb
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

df_train=df_train.dropna(subset=['cycle'])
df_train.loc[df_train['country']==0,'country']=779410
# df_train['country'].unique()



# feature_num=['mileage','transferCount','seatings','displacement','gearbox','newprice','cycle']
# feature_class=['brand','serial','model','color','cityId','carCode','country','maketype','modelyear','oiltype']
# 数值型的相关系数
# data_1=df_train[feature_num]
# pearson=data_1.corr()
# pearson['cycle'].abs().sort_values(ascending=False)
# pearson.index.to_list()


# df_train.dropna(how='any',subset=['carCode','gearbox'],inplace=True)  # 训练集中有缺失值，缺失值较少 ：6个，1个

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
异常值检验
"""
df_train.isnull().sum()

def replace(x):
    QU = x.quantile(0.75)
    QL = x.quantile(0.25)
    IQR = QU -QL
    x[(x>(QU+1.5*IQR)) | (x<(QL - 1.5*IQR))] = np.nan
    return x


nn=['mileage','transferCount','seatings','displacement','gearbox','newprice','pushPrice']
for i in nn:
    df_train[i] = replace(df_train[i])
"""
特征工程
"""
def deal_time(df_train):
    df_train['old_year']=df_train['tradeTime']-df_train['registerDate']
    df_train['old_year']=df_train['old_year'].apply(lambda x:str(x).split(' ')[0])
    df_train['old_year']=df_train['old_year'].astype(int)

    df_train['old_year_1']=df_train['tradeTime']-df_train['licenseDate']
    df_train['old_year_1']=df_train['old_year_1'].apply(lambda x:str(x).split(' ')[0])
    df_train['old_year_1']=df_train['old_year_1'].astype(int)
    return df_train

df_train=deal_time(df_train)
df_test=deal_time(df_test)

def f(x):
    if re.findall(': ".+?"',str(x)):
        res=re.findall(': ".+?"',str(x))[-1]
        a=res.replace('"','').replace(':','').replace(' ','')
        a=float(a)
    else:
        a=0.0
    return a

# res=re.findall(': ".+?"','{"2020-12-29": "17.68", "2020-12-30": "17.8799", "2021-01-15": "16.8799"}')
# a=res[-1].replace('"','').replace(':','').replace(' ','')
# a=float(a)
# res=re.findall(': ".+?"','{}')[-1]
"""
求出最终的价格
"""

df_train['last_updatePrice']=df_train['updatePriceTimeJson'].map(f)
df_train.loc[df_train['last_updatePrice']==0.0000,'last_updatePrice']=df_train.loc[df_train['last_updatePrice']==0.0000,'pushPrice']

df_test['last_updatePrice']=df_test['updatePriceTimeJson'].map(f)
df_test.loc[df_test['last_updatePrice']==0.0000,'last_updatePrice']=df_test.loc[df_test['last_updatePrice']==0.0000,'pushPrice']

"""
构造特征
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

    # df_train=fill_v(df_train)
    df_train['pushDate_year'] = df_train['pushDate'].dt.year
    df_train['pushDate_month'] = df_train['pushDate'].dt.month
    df_train['pushDate_day'] = df_train['pushDate'].dt.day
    return df_train

df_train=deal_ano_feature(df_train)
df_test=deal_ano_feature(df_test)
# 将cycle调整到最后一列

# 数据分箱
bin = [i*5 for i in range(9)]

df_train['pushPrice_bin'] = pd.cut(df_train['pushPrice'], bin, labels=False)

df_test['pushPrice_bin'] = pd.cut(df_test['pushPrice'], bin, labels=False)


df_train=df_train.select_dtypes(include=['int64','float64','int32'])
df_test=df_test.select_dtypes(include=['int64','float64','int32'])

# 对类别特征进行 OneEncoder
# df_train=pd.get_dummies(df_train,columns=['brand','serial','model','pushPrice_bin',
#                                           'color','cityId','carCode','country','maketype','modelyear','oiltype'])
# df_test=pd.get_dummies(df_test,columns=['brand','serial','model','pushPrice_bin',
#                                           'color','cityId','carCode','country','maketype','modelyear','oiltype'])

cycle=df_train['cycle']
df_train=df_train.drop('cycle',axis=1)
df_train['cycle']=cycle

df_train['seatings'].unique()
df_train=df_train.drop('seatings',axis=1)

pearson=df_train.corr()
index = pearson['cycle'][:-1].abs() > 0.03
X_submit=df_train.iloc[:,:-1]
X_submit=X_submit.loc[:,index]
features=X_submit.columns

y=np.log1p(df_train['cycle'])

X_train,X_test,y_train,y_test=train_test_split(X_submit.iloc[:,:],y,test_size=0.3)


clf=lgb.LGBMRegressor(
    learning_rate=0.01,
    max_depth=-1,
    n_estimators=5000,
    boosting_type='gbdt',
    random_state=2022,
    objective='regression',
)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_pred=np.expm1(y_pred)
res=mean_absolute_error(y_test,y_pred)
print("lightGbm: ",res)
# 13.206856546563632
X_=df_test[features]
X_.columns
y_res=clf.predict(X_)
y_res=np.expm1(y_res)
df_test['cycle']=y_res
data=df_test[['carid','cycle']]
data.to_csv(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\附件6：门店交易模型结果.txt',sep='\t',index=None,header=None)



df_test


