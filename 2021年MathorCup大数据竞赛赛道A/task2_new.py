import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df_1=pd.read_table(r'D:\比赛\2021年MathorCup大数据竞赛赛道A\附件\附件4：门店交易训练数据.txt',sep='\t',
                 parse_dates=[1,4,5],header=None,names=['carid','pushDate','pushPrice','updatePriceTimeJson'
                ,'pullDate','withdrawDate'])
data=pd.DataFrame(data=df_1)
data['pushDate_year']=data['pushDate'].dt.year
data['pushDate_month']=data['pushDate'].dt.month
data['pushDate_day']=data['pushDate'].dt.day

data['withdrawDate_year']=data['withdrawDate'].dt.year
data['withdrawDate_month']=data['withdrawDate'].dt.month
data['withdrawDate_day']=data['withdrawDate'].dt.day


data_1=data
data_1['period']=data_1['withdrawDate']-data_1['pushDate']
# pd.to_datetime(data['period'])  dtype timedelta64[ns] cannot be converted to datetime64[ns]
data_1['period']=data_1['period'].apply(lambda x: str(x).split(' ')[0])
data_1['period']=data_1['period'].apply(lambda x: int(x) if x!='NaT' else np.nan)

data_1['updatePriceTimeJson'].astype('string')
def deal(data=None):
    li=[]
    for i in data.index:
        count=len(re.findall('\d+-\d+-\d+',str(data.loc[i,'updatePriceTimeJson'])))
        li.append(count)

    return li

data_1['decreasing_count']=deal(data_1)

data_1.info()

data_2=data_1.drop(['pushDate','pullDate','withdrawDate','updatePriceTimeJson'],axis=1)
data_2=data_2.dropna()

y_=data_2['period']
data_2=data_2.drop(['period'],axis=1)
data_2['period']=y_
data_2.info()


pearson=data_2.corr()
index = pearson['period'][:-1].abs() > 0.01
X=data_2.iloc[:,:-1]
X_submit=X.loc[:,index]
X_submit.columns
X_train,X_test,y_train,y_test=train_test_split(X_submit,data_2['period'],test_size=0.2)

random_model=RandomForestRegressor(n_estimators=300,random_state=33,n_jobs=-1)
random_model.fit(X_train,y_train)
y_pred=random_model.predict(X_test)
random_model.score(X_test,y_test)