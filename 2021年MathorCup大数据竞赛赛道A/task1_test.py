import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def train(data_0=None):
    data=pd.DataFrame(data=data_0)
    columns = ['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode',
               'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear',
               'displacement', 'gearbox', 'oiltype', 'newprice']
    for i in range(1, 16):
        str_ = 'anonymousFeature' + str(i)
        columns.append(str_)
    columns.append('price')
    data.columns = columns

    # data=data.drop(['anonymousFeature4','anonymousFeature7','anonymousFeature10','anonymousFeature15'],axis=1)  # 删除缺失值严重的特征
    data = data.drop(['country', 'anonymousFeature4', 'anonymousFeature7', 'anonymousFeature10', 'anonymousFeature15'
                         , 'maketype', 'anonymousFeature1', 'anonymousFeature8', 'anonymousFeature9'], axis=1)
    data.info()
    nn = ['carCode', 'modelyear', 'gearbox']

    for i in nn:
        x = int(data[i].mode())
        data[i].fillna(x, inplace=True)


    """
    特征构造
    """
    data['tradeTime_year'] = data['tradeTime'].dt.year
    data['tradeTime_month'] = data['tradeTime'].dt.month
    data['tradeTime_day'] = data['tradeTime'].dt.day
    data['registerDate_year'] = data['registerDate'].dt.year
    data['registerDate_month'] = data['registerDate'].dt.month
    data['registerDate_day'] = data['registerDate'].dt.month

    print(data['registerDate_day'].value_counts().plot(kind='bar', rot=30))
    plt.show()

    # 处理定类数据（Frequency编码）
    data['color'] = data['color'].map(data['color'].value_counts())
    data['carCode'] = data['carCode'].map(data['carCode'].value_counts())
    # data['country'] = data['country'].map(data['country'].value_counts())
    data['modelyear'] = data['modelyear'].map(data['modelyear'].value_counts())

    # 处理匿名特征11
    data_1 = data.dropna()  #
    data_1.info()
    len(list(data_1['anonymousFeature11']))


    def deal_11(x):
        return sum([float(x) for x in re.findall("\d", x)])

    data_1['anonymousFeature11'] = data_1['anonymousFeature11'].map(deal_11)
    data_1.info()

    # 处理匿名特征12
    def deal_12(x):
        li = [float(x) for x in re.findall("\d+", x)]
        return li[0] * li[1] * li[2]

    data_1['anonymousFeature12_length'] = data_1['anonymousFeature12'].apply(lambda x: int(x.split('*')[0]))
    data_1['anonymousFeature12_width'] = data_1['anonymousFeature12'].apply(lambda x: int(x.split('*')[1]))
    data_1['anonymousFeature12_height'] = data_1['anonymousFeature12'].apply(lambda x: int(x.split('*')[2]))
    data_1['anonymousFeature12'] = data_1['anonymousFeature12'].map(deal_12)

    # 处理匿名特征13
    def deal_13(x):
        return x[:4], x[4:6]

    data_1['anonymousFeature13'] = data_1['anonymousFeature13'].astype('string')
    data_1['anonymousFeature13_year'] = data_1['anonymousFeature13'].map(deal_13)  # (2017, 09)
    data_1['anonymousFeature13_month'] = data_1['anonymousFeature13_year'].apply(lambda x: int(x[1]))
    data_1['anonymousFeature13_year'] = data_1['anonymousFeature13_year'].apply(lambda x: int(x[0]))
    data_1['anonymousFeature13'] = data_1['anonymousFeature13'].astype('float')


    data_1['old_year'] = data_1['tradeTime'] - data_1['registerDate']
    data_1['old_year'] = data_1['old_year'].apply(lambda x: str(x).split(' ')[0])
    data_1['old_year'] = data_1['old_year'].astype(int)

    data_1['old_year_1'] = data_1['tradeTime'] - data_1['licenseDate']
    data_1['old_year_1'] = data_1['old_year_1'].apply(lambda x: str(x).split(' ')[0])
    data_1['old_year_1'] = data_1['old_year_1'].astype(int)


    # 数据分桶
    bin = [0, 1, 4, 7.15, 10, 50]
    data_1['mileage_bin'] = pd.cut(data_1['mileage'], bins=bin, labels=False)

    y = data_1['price']
    data_2 = data_1.drop(['price'], axis=1)
    print(data_2.columns)
    data_2['price'] = y
    data_2.info()
    # data_2=data_2.dropna()
    data_2 = data_2.drop(['tradeTime', 'registerDate', 'licenseDate'], axis=1)

    # 看看有没有异常值
    data_2.describe()
    data_2 = data_2[data_2['price'] < 80]
    data_2.columns
    data_2.info()
    # data_2.drop([22115],axis=0,inplace=True)
    """
    特征选择
    """
    # 基于皮尔逊相关系数
    pearson = data_2.corr()
    index = pearson['price'][:-1].abs() > 0.1
    X = data_2.iloc[:, :-1]
    X_subset = X.loc[:, index]
    # X_subset.columns

    """
    降维
    """

    """
    模型训练
    """
    # standardScaler = StandardScaler()
    # X = standardScaler.fit_transform(X_subset.to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X_subset, data_2['price'].to_numpy(), test_size=0.2, random_state=3)

    random_model = RandomForestRegressor(n_estimators=500, random_state=33, n_jobs=-1)
    random_model.fit(X_train, y_train)
    y_pred = random_model.predict(X_test)
    return random_model,X_subset.columns,y_pred,y_test
def test(data_0=None,model=None,feature=None):
    data=pd.DataFrame(data=data_0)
    columns = ['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode',
               'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear',
               'displacement', 'gearbox', 'oiltype', 'newprice']
    for i in range(1, 16):
        str_ = 'anonymousFeature' + str(i)
        columns.append(str_)
    data.columns = columns


    # data=data.drop(['anonymousFeature4','anonymousFeature7','anonymousFeature10','anonymousFeature15'],axis=1)  # 删除缺失值严重的特征
    data = data.drop(['country', 'anonymousFeature4', 'anonymousFeature7', 'anonymousFeature10', 'anonymousFeature15'
                         , 'maketype', 'anonymousFeature1', 'anonymousFeature8', 'anonymousFeature9'], axis=1)

    nn = ['carCode', 'modelyear', 'gearbox','anonymousFeature13']
    for i in nn:
        x = int(data[i].mode())
        data[i].fillna(x, inplace=True)


    """
    特征构造
    """
    data['tradeTime_year'] = data['tradeTime'].dt.year
    data['tradeTime_month'] = data['tradeTime'].dt.month
    data['tradeTime_day'] = data['tradeTime'].dt.day
    data['registerDate_year'] = data['registerDate'].dt.year
    data['registerDate_month'] = data['registerDate'].dt.month
    data['registerDate_day'] = data['registerDate'].dt.day



    # 处理定类数据（Frequency编码）
    data['color'] = data['color'].map(data['color'].value_counts())
    data['carCode'] = data['carCode'].map(data['carCode'].value_counts())
    # data['country'] = data['country'].map(data['country'].value_counts())
    data['modelyear'] = data['modelyear'].map(data['modelyear'].value_counts())

    # 处理匿名特征11
    data_1=data
    data_1.info()
    data_1['anonymousFeature11'].fillna('3',inplace=True)
    def deal_11(x):
        return sum([float(x) for x in re.findall("\d", x)])

    data_1['anonymousFeature11'] = data_1['anonymousFeature11'].map(deal_11)
    data_1.info()

    # 处理匿名特征12
    data_1['anonymousFeature12'].fillna('5015*1874*1455',inplace=True)
    def deal_12(x):
        li = [float(x) for x in re.findall("\d+", x)]
        return li[0] * li[1] * li[2]

    data_1['anonymousFeature12_length'] = data_1['anonymousFeature12'].apply(lambda x: int(x.split('*')[0]))
    data_1['anonymousFeature12_width'] = data_1['anonymousFeature12'].apply(lambda x: int(x.split('*')[1]))
    data_1['anonymousFeature12_height'] = data_1['anonymousFeature12'].apply(lambda x: int(x.split('*')[2]))
    data_1['anonymousFeature12'] = data_1['anonymousFeature12'].map(deal_12)

    # 处理匿名特征13
    def deal_13(x):
        return x[:4], x[4:6]

    data_1['anonymousFeature13'] = data_1['anonymousFeature13'].astype('string')
    data_1['anonymousFeature13_year'] = data_1['anonymousFeature13'].map(deal_13)  # (2017, 09)
    data_1['anonymousFeature13_month'] = data_1['anonymousFeature13_year'].apply(lambda x: int(x[1]))
    data_1['anonymousFeature13_year'] = data_1['anonymousFeature13_year'].apply(lambda x: int(x[0]))
    data_1['anonymousFeature13'] = data_1['anonymousFeature13'].astype('float')


    data_1['old_year'] = data_1['tradeTime'] - data_1['registerDate']
    data_1['old_year'] = data_1['old_year'].apply(lambda x: str(x).split(' ')[0])
    data_1['old_year'] = data_1['old_year'].astype(int)

    data_1['old_year_1'] = data_1['tradeTime'] - data_1['licenseDate']
    data_1['old_year_1'] = data_1['old_year_1'].apply(lambda x: str(x).split(' ')[0])
    data_1['old_year_1'] = data_1['old_year_1'].astype(int)



    # 数据分桶
    bin = [0, 1, 4, 7.15, 10, 50]
    data_1['mileage_bin'] = pd.cut(data_1['mileage'], bins=bin, labels=False)
    data_2 = data_1.drop(['tradeTime', 'registerDate', 'licenseDate'], axis=1)
    X=data_2[feature]
    y=model.predict(X)
    return X ,y

df = pd.read_table(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\附件\附件1：估价训练数据.txt',
                    parse_dates=[1, 11, 12],sep='\t', header=None, encoding='gbk')
df_1 = pd.read_table(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\附件\附件2：估价验证数据.txt',
                    parse_dates=[1, 11, 12],sep='\t', header=None, encoding='gbk')

model,feature,y_p,y_t=train(data_0=df)
X, y_=test(data_0=df_1,model=model,feature=feature)
data_=pd.DataFrame(df_1.iloc[:,0])
data_['price']=y_
print(data_)
data_.to_csv('../data/附件3：估价模型结果.txt',sep='\t',index=False, header=None)
# data_.to_csv(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\附件3：估价模型结果.txt',sep='\t',index=False, header=None)
