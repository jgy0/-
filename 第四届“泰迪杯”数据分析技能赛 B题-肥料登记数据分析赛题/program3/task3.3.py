import pandas as pd

da = pd.read_excel('./data/附件3.xlsx')

# 1. 统计产品登记数量大于10的公司
qiye = list(da['企业名称'])

def counter(list):
    from collections import Counter
    # 对列表内的元素进行排序，计数，以字典的形式返回
    list_counter = Counter(list)
    list_sorted = sorted(list_counter.items(), key = lambda x:x[1], reverse = True)
    list_dict = dict(list_sorted)
    return list_dict

dqiye = counter(qiye)
# print(dqiye)

hqiye = []
for co in dqiye:
    if dqiye[co] > 10:
        hqiye.append(co)

df = da.loc[da['企业名称'].isin(hqiye)]
# print(df)

# 统计企业所用到的肥料的集合
def qytong(df, str):
    df_1 = df[df['企业名称'] == str]
    df_1 = df_1.drop(columns = ['序号', '企业名称', '发酵菌剂'])
    df_non = df_1.count(axis = 0)
    feiliao = list(df_non.index)
    count = 0
    fliao = []
    for co in feiliao:
        if df_non[co] > 0:
            count = count + 1
            fliao.append(co)
    return fliao


fliao_1= qytong(df, 'ID1')
# print(fliao_1)
# print(count_1)
dd = {}
for co in hqiye:
    fliao_1= qytong(df, co)
    dd[co] = fliao_1
# print(dd)
# print(len(dd))

data = pd.DataFrame(columns = hqiye, index = hqiye)

for i in hqiye:
    for j in hqiye:
        ifliao = dd[i]
        jfliao = dd[j]
        # print(ifliao)
        # print(jfliao)
        num1 = 0
        for a in ifliao:
            if a in jfliao:
                num1 = num1 + 1
        num2 = len(ifliao)
        for b in jfliao:
            if b not in ifliao:
                num2 = num2 + 1
        data.loc[i, j] = num1 / num2

data.to_excel('./result/result3_3.xlsx')
