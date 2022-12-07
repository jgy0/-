import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures      # 增加特征
import mpl_toolkits.mplot3d
import pyecharts.options as opts
from pyecharts.charts import Radar

df = pd.read_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\附件2.xlsx')
df['含磷百分比'] = df['P2O5百分比'].apply(lambda x: x*(62/152))
df['含钾百分比'] = df['K2O百分比'].apply(lambda x: x*(78/94))

data = df[['总氮百分比', '含磷百分比', '含钾百分比']]
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
poly_X = poly.fit_transform(data)

model = KMeans(n_clusters=4).fit(poly_X)
# 类中心个数   n_clusters
y = pd.DataFrame(model.labels_)
y=y.apply(lambda x: x+1)
data['label'] = y  # 就是聚类的结果 model.labels_
# print(data['label'].unique())
# metrics模块 可以进行模型的评估
df['聚类标签'] = data['label']
df = df.drop(['含磷百分比', '含钾百分比'],axis=1)

df.to_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\result\result2_3.xlsx',index=False)
"""
绘制一维散点图
"""
d = data[data['label'] == 1]
plt.plot(d['含钾百分比'], d['含磷百分比'], 'r.')
d = data[data['label'] == 2]
plt.plot(d['含钾百分比'], d['含磷百分比'], 'go')
d = data[data['label'] == 3]
plt.plot(d['含钾百分比'], d['含磷百分比'], 'b*')
d = data[data['label'] == 4]
plt.plot(d['含钾百分比'], d['含磷百分比'], 'yH')
plt.legend(['1', '2', '3', '4'])
# data.info()
# plt.scatter(list(data['FundMoney']),list(data['Surplus']),s=50,cmap='rainbow')
plt.show()

"""
三维散点图
"""
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data_1 = data[data['label'] == 1]
data_2 = data[data['label'] == 2]
data_3 = data[data['label'] == 3]
data_4 = data[data['label'] == 4]



ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
ax.set_title('3d_image_show')  # 设置本图名称

ax.scatter(data_1['总氮百分比'], data_1['含磷百分比'], data_1['含钾百分比'], c='r')   # 绘制数据点 c: 'r'红色，'y'黄色，
ax.scatter(data_2['总氮百分比'], data_2['含磷百分比'], data_2['含钾百分比'], c='y')
ax.scatter(data_3['总氮百分比'], data_3['含磷百分比'], data_3['含钾百分比'], c='b')
ax.scatter(data_4['总氮百分比'], data_4['含磷百分比'], data_4['含钾百分比'], c='#2ca02c')



ax.set_xlabel('X')  # 设置x坐标轴
ax.set_ylabel('Y')  # 设置y坐标轴
ax.set_zlabel('Z')  # 设置z坐标轴
ax.legend(['类型一', '类型二', '类型三', '类型四'])
plt.show()
"""
绘制散点图矩阵
"""
n = [data_1, data_2, data_3, data_4]
for X in n:
    df = X[['总氮百分比', '含磷百分比', '含钾百分比']]
    pd.plotting.scatter_matrix(df)
    plt.show()


"""
绘制雷达图
"""
v1 = [[data_1['总氮百分比'].mean(), data_1['含磷百分比'].mean(), data_1['含钾百分比'].mean()]]
v2 = [[data_2['总氮百分比'].mean(), data_2['含磷百分比'].mean(), data_2['含钾百分比'].mean()]]
v3 = [[data_3['总氮百分比'].mean(), data_3['含磷百分比'].mean(), data_3['含钾百分比'].mean()]]
v4 = [[data_4['总氮百分比'].mean(), data_4['含磷百分比'].mean(), data_4['含钾百分比'].mean()]]


(
    Radar(init_opts=opts.InitOpts(width="1280px", height="720px", bg_color="#CCCCCC"))
    .add_schema(
        schema=[
            opts.RadarIndicatorItem(name="总氮百分比"),
            opts.RadarIndicatorItem(name="含磷百分比"),
            opts.RadarIndicatorItem(name="含钾百分比"),
        ],
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
        ),
        textstyle_opts=opts.TextStyleOpts(color="#fff"),
    )
    .add(
        series_name="类型一",
        data=v1,
        linestyle_opts=opts.LineStyleOpts(color="#CD0000"),
    )
    .add(
        series_name="类型二",
        data=v2,
        linestyle_opts=opts.LineStyleOpts(color="#5CACEE"),
    )
    .add(
        series_name="类型三",
        data=v3,
        linestyle_opts=opts.LineStyleOpts(color='rgb(128, 128, 128)'),
    )
        .add(
        series_name="类型四",
        data=v4,
        linestyle_opts=opts.LineStyleOpts(color='rgb(255, 255, 255)'),
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="类别雷达图"), legend_opts=opts.LegendOpts()
    )
    .render(r"D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\result\basic_radar_chart.html")
)