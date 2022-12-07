import pandas as pd

df = pd.read_excel('./data/附件2.xlsx')
# print(df)

dfy = df[df['产品通用名称'] == '有机肥料']
# print(dfy)

def fenzu(df1, str1):
    yang_max = df1[str1].max()
    yang_min = df1[str1].min()
    # print(yang_max)
    # print(yang_min)

    dd = (yang_max - yang_min) / 10  # 间距
    # print(dd)
    zubie = []
    # print(df1[str1])
    for co in df1[str1]:
        if co <= (yang_min + dd):
            # print(co)
            zubie.append(1)
        elif co <= (yang_min + (2 * dd)):
            zubie.append(2)
        elif co <= (yang_min + (3 * dd)):
            zubie.append(3)
        elif co <= (yang_min + (4 * dd)):
            zubie.append(4)
        elif co <= (yang_min + (5 * dd)):
            zubie.append(5)
        elif co <= (yang_min + (6 * dd)):
            zubie.append(6)
        elif co <= (yang_min + (7 * dd)):
            zubie.append(7)
        elif co <= (yang_min + (8 * dd)):
            zubie.append(8)
        elif co <= (yang_min + (9 * dd)):
            zubie.append(9)
        else:
            zubie.append(10)
    return zubie

zuwuji = fenzu(dfy, '总无机养分百分比')
zuyouji = fenzu(dfy, '有机质百分比')

zubie = list(zip(zuwuji, zuyouji))
# print(len(zubie))

dfy['分组标签'] = zubie
dfy.to_excel('./result/result2_2.xlsx', index=False)


# 组别的第一个序号为总无机养分百分比，第二个为有机质百分比分组
zu = list(dfy['分组标签'])
# print(zu)

def counter(list):
    from collections import Counter
    # 对列表内的元素进行排序，计数，以字典的形式返回
    list_counter = Counter(list)
    list_sorted = sorted(list_counter.items(), key = lambda x:x[1], reverse = True)
    list_dict = dict(list_sorted)
    return list_dict


wuji = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
youji = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
dzu = counter(zu)
# print(dzu)
dd = []
for co in dzu:
    dl = []
    b = str(co)
    a = list(b.split(', '))
    dl.append(a[0][1:])
    dl.append(a[1][:-1])
    dl.append(dzu[co])
    # print(dl)
    dd.append(dl)

from pyecharts import options as opts
from pyecharts.charts import HeatMap
c = (
    HeatMap()
    .add_xaxis(wuji)
    .add_yaxis(
        "登记数量",
        youji,
        dd,
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="有机肥料产品热力图"),
        visualmap_opts=opts.VisualMapOpts(),
    )

    .render("./result/有机材料产品热力图2-2.html")
)