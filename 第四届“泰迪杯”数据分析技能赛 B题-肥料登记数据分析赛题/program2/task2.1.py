import pandas as pd

df = pd.read_excel('./data/附件2.xlsx')


dffu = df[df['产品通用名称'] == '复混肥料']


yang_max = df['总无机养分百分比'].max()
yang_min = df['总无机养分百分比'].min()

# 总无机养分百分比最大为0.72，最小为0.0，将其分为10组
dd = (yang_max - yang_min) /10  # 间距


zubie = []
for co in dffu['总无机养分百分比']:
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

dffu['总无机养分分组标签'] = zubie

# 生成xlsx文件
# dffu.to_excel('./result/result2_1.xlsx', index=False)

dffu_1 = dffu[dffu['总无机养分分组标签'] == 1]
# print(dffu_1)
# print(len(dffu_1))
dffu_2 = dffu[dffu['总无机养分分组标签'] == 2]
dffu_3 = dffu[dffu['总无机养分分组标签'] == 3]
dffu_4 = dffu[dffu['总无机养分分组标签'] == 4]
dffu_5 = dffu[dffu['总无机养分分组标签'] == 5]
dffu_6 = dffu[dffu['总无机养分分组标签'] == 6]
dffu_7 = dffu[dffu['总无机养分分组标签'] == 7]
dffu_8 = dffu[dffu['总无机养分分组标签'] == 8]
dffu_9 = dffu[dffu['总无机养分分组标签'] == 9]
dffu_10 = dffu[dffu['总无机养分分组标签'] == 10]

dzu = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
dlen = [len(dffu_1), len(dffu_2), len(dffu_3), len(dffu_4), len(dffu_5), len(dffu_6), len(dffu_7), len(dffu_8), len(dffu_9), len(dffu_10)]
# print(dlen)

# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
# [3, 0, 0, 373, 1154, 1470, 2098, 841, 14, 1]

# 绘制直方图
from pyecharts import options as opts
from pyecharts.charts import Bar
c = (
    Bar()
    .add_xaxis(dzu)
    .add_yaxis("登记数量", dlen)
    .set_global_opts(title_opts=opts.TitleOpts(title="复混肥料产品分布"))
    .render("./result/复混材料产品分布直方图2-1.html")
)