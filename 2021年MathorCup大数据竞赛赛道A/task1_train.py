import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import xgboost as xgb

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""
评估函数
"""
def estimate(y_true=None,y_pred=None):
    y_true=np.array(list(y_true))
    y_pred=np.array(list(y_pred))
    Ape=np.abs(y_pred-y_true)/y_true
    Mape=sum(Ape)/len(y_true)
    # Ape_count=[np.nan if x <=0.05 else x for x in Ape]
    Ape_count=len(np.where(Ape<=0.05)[0])/len(Ape)
    return 0.2*(1-Mape)+0.8*Ape_count

"""
读入数据和数据清洗
"""
df = pd.read_table(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\附件\附件1：估价训练数据.txt',
                    parse_dates=[1, 11, 12],sep='\t', header=None, encoding='gbk')
# df = pd.read_table('../data/附件1：估价训练数据.txt', sep='\t', header=None, encoding='gbk')
data = pd.DataFrame(data=df)
columns = ['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode',
           'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear',
           'displacement', 'gearbox', 'oiltype', 'newprice']
for i in range(1, 16):
    str_ = 'anonymousFeature'+str(i)
    columns.append(str_)
columns.append('price')
data.columns = columns

data=data.drop(['country','anonymousFeature4','anonymousFeature7','anonymousFeature10','anonymousFeature15'
                ,'maketype','anonymousFeature1','anonymousFeature8','anonymousFeature9'],axis=1) # 删除缺失值严重的特征
data.info()
nn = ['carCode', 'modelyear', 'gearbox']

for i in nn:
    x = int(data[i].mode())
    data[i].fillna(x, inplace=True)

"""
特征构造
"""
data['tradeTime_year']=data['tradeTime'].dt.year
data['tradeTime_month']=data['tradeTime'].dt.month
data['tradeTime_day']=data['tradeTime'].dt.day
data['registerDate_year']=data['registerDate'].dt.year
data['registerDate_month']=data['registerDate'].dt.month
data['registerDate_day']=data['registerDate'].dt.day

print(data['registerDate_day'].value_counts().plot(kind='bar',rot=30))
plt.show()

# 处理定类数据（Frequency编码）
data['color'] = data['color'].map(data['color'].value_counts())
data['carCode'] = data['carCode'].map(data['carCode'].value_counts())
data['modelyear'] = data['modelyear'].map(data['modelyear'].value_counts())

# 处理匿名特征11
data_1=data.dropna()                #
data_1.info()

def deal_11(x):
    return sum([float(x) for x in re.findall("\d",x)])
data_1['anonymousFeature11']=data_1['anonymousFeature11'].map(deal_11)
data_1.info()
# 处理匿名特征12
def deal_12(x):
    li=[float(x) for x in re.findall("\d+",x)]
    return li[0]*li[1]*li[2]
data_1['anonymousFeature12_length']=data_1['anonymousFeature12'].apply(lambda x:int(x.split('*')[0]))
data_1['anonymousFeature12_width']=data_1['anonymousFeature12'].apply(lambda x:int(x.split('*')[1]))
data_1['anonymousFeature12_height']=data_1['anonymousFeature12'].apply(lambda x:int(x.split('*')[2]))
data_1['anonymousFeature12']=data_1['anonymousFeature12'].map(deal_12)
# 处理匿名特征13
def deal_13(x):
    return x[:4], x[4:6]
data_1['anonymousFeature13']=data_1['anonymousFeature13'].astype('string')
data_1['anonymousFeature13_year']=data_1['anonymousFeature13'].map(deal_13)    #  (2017, 09)
data_1['anonymousFeature13_month']=data_1['anonymousFeature13_year'].apply(lambda x: int(x[1]))
data_1['anonymousFeature13_year']=data_1['anonymousFeature13_year'].apply(lambda x: int(x[0]))
data_1['anonymousFeature13']=data_1['anonymousFeature13'].astype('float')


# 构建新特征
data_1['old_year']=data_1['tradeTime']-data_1['registerDate']
data_1['old_year']=data_1['old_year'].apply(lambda x:str(x).split(' ')[0])
data_1['old_year']=data_1['old_year'].astype(int)

data_1['old_year_1']=data_1['tradeTime']-data_1['licenseDate']
data_1['old_year_1']=data_1['old_year_1'].apply(lambda x:str(x).split(' ')[0])
data_1['old_year_1']=data_1['old_year_1'].astype(int)


# 数据分桶
bin=[0, 1, 4, 7.15, 10, 50]
data_1['mileage_bin']=pd.cut(data_1['mileage'],bins=bin,labels=False)

y = data_1['price']
data_2 = data_1.drop(['price'],axis=1)
print(data_2.columns)
data_2['price'] = y
data_2.info()
# data_2=data_2.dropna()
data_2=data_2.drop(['tradeTime','registerDate','licenseDate'],axis=1)

# 看看有没有异常值
data_2.describe()
data_2=data_2[data_2['price']<80]
data_2.columns
data_2.info()

"""
特征选择
"""
# 基于皮尔逊相关系数
pearson = data_2.corr()
index = pearson['price'][:-1].abs() > 0.1
X = data_2.iloc[:,:-1]
X_subset = X.loc[:, index]
# X_subset.columns

"""
降维
"""
# pca=PCA(n_components=6)
# X_new_sk=pca.fit_transform(X_subset)


"""
模型训练
"""
# standardScaler=StandardScaler()
# X = standardScaler.fit_transform(X_subset)

X_train, X_test, y_train, y_test = train_test_split(X_subset,data_2['price'].to_numpy() , test_size=0.2,random_state=3)

random_model = RandomForestRegressor(n_estimators=500,random_state=33,n_jobs=-1)
random_model.fit(X_train,y_train)
y_pred = random_model.predict(X_test)
score = estimate(y_true=y_test,y_pred=y_pred)

"""
XGboost优化模型
"""
xlf = xgb.XGBRegressor(max_depth=50,  # 差不多
                        learning_rate=0.05,
                        n_estimators=300,
                        min_child_weight=2,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        seed=0,
                        )
xlf.fit(X_train, y_train, eval_metric='mae', eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=20)
preds = xlf.predict(X_test)
score=estimate(y_true=y_test,y_pred=preds)
score

"""
画散点图观察模型的效果
"""
random_model.score(X_test,y_test)
random_model.score(X_train,y_train)
np.array(y_test)
plt.scatter(X_test['newprice'], y_test)  # 样本实际分布
plt.scatter(X_test['newprice'], y_pred, color='red')   # 绘制拟合曲线
plt.legend(['实际值','预测值'])
plt.xlabel("newprice")
plt.ylabel("price")
plt.show()















