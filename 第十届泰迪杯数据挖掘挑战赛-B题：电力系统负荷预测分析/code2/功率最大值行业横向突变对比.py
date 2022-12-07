import pyecharts.options as opts
from pyecharts.charts import Line

"""
Gallery 使用 pyecharts 1.1.0
参考地址: https://www.echartsjs.com/examples/editor.html?c=line-log

目前无法实现的功能:

1、暂无
"""

import pandas as pd
#data1 = pd.read_csv(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最大值.csv")
data1 = pd.read_csv(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最小值.csv")
#print(data1)
x = data1["Time"]
#x_data = [d.replace(" ", "-") for d in x]
y_max1 = data1["商业"]
y_max2 = data1["大工业用电"]
y_max3 = data1["普通工业"]
y_max4 = data1["非普工业"]

#data2 = pd.read_csv(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最小值.csv")
#y_min = data2["商业"]

(
    Line(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add_xaxis(xaxis_data=x)
    .add_yaxis(
        series_name="商业有功功率最小值",
        y_axis=y_max1,
        linestyle_opts=opts.LineStyleOpts(width=2),
    )
    .add_yaxis(
        series_name="大工业用电有功功率最小值", y_axis=y_max2, linestyle_opts=opts.LineStyleOpts(width=2)
    )
    .add_yaxis(
        series_name="普通工业有功功率最小值", y_axis=y_max3, linestyle_opts=opts.LineStyleOpts(width=2)
    )
    .add_yaxis(
        series_name="非普工业有功功率最小值", y_axis=y_max4, linestyle_opts=opts.LineStyleOpts(width=2)
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="功率最小值行业对比", pos_left="center"),
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}"),
        legend_opts=opts.LegendOpts(pos_left="left"),
        xaxis_opts=opts.AxisOpts(type_="category", name="x"),
        yaxis_opts=opts.AxisOpts(
            #type_="log",
            name="y",
            splitline_opts=opts.SplitLineOpts(is_show=True),
            is_scale=True,
        ),
    )
    .render("log_axis.html")
)
