import pandas as pd
import re

df = pd.read_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\附件4.xlsx')
df[['总氮百分比', 'P2O5百分比', 'K2O百分比', '有机质百分比']] = 0

df.loc[df['产品通用名称'] == '复混肥料', '有机质百分比'] = 0
# re.findall('^\d_\d+\.jpg$', i) != []:

df['技术指标'] = list(map(str, list(df['技术指标'])))

def get_names(x):
    li=[]
    for i in x:
        if re.findall('氯', i) == []:
            li.append('无氯')
        elif re.findall('含氯', i) !=[]:
            li.append('低氯')
        elif re.findall('高氯',i) !=[]:
            li.append('高氯')
        elif re.findall('中氯',i) !=[]:
            li.append('中氯')
        elif re.findall('低氯',i) !=[]:
            li.append('低氯')
        else:
            li.append('无氯')
    return li


def get_percent(x):
    for i in x.index:
        if re.findall('总养分',df.loc[i,'技术指标']) !=[]:
            count = re.findall('\d\d', df.loc[i, '技术指标'])[0]
            # print(count)
            count = int(count)
            df.loc[i, ['总氮百分比', 'P2O5百分比', 'K2O百分比']] = round(count/300, 3)
            df.loc[i, '有机质百分比'] = round(count/100, 3)
        elif re.findall('P2O5',df.loc[i,'技术指标']) !=[]:
            count = re.findall('\d\d', df.loc[i, '技术指标'])[0]
            count = int(count)/100
            df.loc[i, '总氮百分比'] = round(count*(14/260), 3)
            df.loc[i, 'P2O5百分比'] = round(count * (152 / 260), 3)
            df.loc[i, 'K2O百分比'] = round(count * (94 / 260), 3)
        else:
            pass



get_percent(df)




df['含氯情况'] = get_names(list(df['技术指标']))
df=df.drop(['技术指标', '原料与占比'], axis=1)
#
# df = df[(df['技术指标'] != '20-0-5') & (df['技术指标'] != pd.to_datetime('2017/8/15 0:00:00'))]

df.to_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\result\result4_1.xlsx', index=False)


