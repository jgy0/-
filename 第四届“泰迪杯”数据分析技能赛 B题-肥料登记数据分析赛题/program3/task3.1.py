import pandas as pd

df = pd.read_excel('./result/result2_1.xlsx')

# print(type(df['发证日期']))
df['年份'] = df['发证日期'].apply(lambda x: x[:4])
# print(df)

df_1 = df[df['组别'] == 1]
# print(df_1)
# print(len(df_1))
df_2 = df[df['组别'] == 2]
df_3 = df[df['组别'] == 3]
df_4 = df[df['组别'] == 4]
df_5 = df[df['组别'] == 5]
df_6 = df[df['组别'] == 6]
df_7 = df[df['组别'] == 7]
df_8 = df[df['组别'] == 8]
df_9 = df[df['组别'] == 9]
df_10 = df[df['组别'] == 10]

def counter(list):
    from collections import Counter
    # 对列表内的元素进行排序，计数，以字典的形式返回
    list_counter = Counter(list)
    list_sorted = sorted(list_counter.items(), key = lambda x:x[1], reverse = True)
    list_dict = dict(list_sorted)
    return list_dict

nian_1 = list(df_1['年份'])
nian_2 = list(df_2['年份'])
nian_3 = list(df_3['年份'])
nian_4 = list(df_4['年份'])
nian_5 = list(df_5['年份'])
nian_6 = list(df_6['年份'])
nian_7 = list(df_7['年份'])
nian_8 = list(df_8['年份'])
nian_9 = list(df_9['年份'])
nian_10 = list(df_10['年份'])

# print(set(list(df['年份'])))
# 复混肥料发证日期的年份为2012-2020
# nian = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']

def tongji(l):
    dn = counter(l)
    dnn1 = {'2012': 0, '2013': 0, '2014': 0, '2015': 0, '2016': 0, '2017': 0, '2018': 0, '2019': 0, '2020': 0}
    for co in dn:
        if co in dnn1:
            dnn1[co] = dn[co]
    return dnn1

dn1 = tongji(nian_1)
dn2 = tongji(nian_2)
dn3 = tongji(nian_3)
dn4 = tongji(nian_4)
dn5 = tongji(nian_5)
dn6 = tongji(nian_6)
dn7 = tongji(nian_7)
dn8 = tongji(nian_8)
dn9 = tongji(nian_9)
dn10 = tongji(nian_10)

zu = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
nian = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']

print(dn1)

import pyecharts.options as opts
from pyecharts.charts import Line
c = (
    Line()
    .add_xaxis(xaxis_data=nian)
    .add_yaxis(series_name="1",
                y_axis=list(dn1.values()),
            )
    .add_yaxis(series_name="2",
                y_axis=list(dn2.values()),
            )
    .add_yaxis(series_name="3",
                y_axis=list(dn3.values()),
            )
    .add_yaxis(series_name="4",
                y_axis=list(dn4.values()),
            )
    .add_yaxis(series_name="5",
                y_axis=list(dn5.values()),
            )
    .add_yaxis(series_name="6",
                y_axis=list(dn6.values()),
            )
    .add_yaxis(series_name="7",
                y_axis=list(dn7.values()),
            )
    .add_yaxis(series_name="8",
                y_axis=list(dn8.values()),
            )
    .add_yaxis(series_name="9",
                y_axis=list(dn9.values()),
            )
    .add_yaxis(series_name="10",
                y_axis=list(dn10.values()),
           )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="各组别不同年份产品等级数量变化")
    )
    .render('./result/各组别不同年份产品分等级数量变化趋势3-1.html')
)






