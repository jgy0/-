import pyecharts.options as opts
from pyecharts.charts import Line
import pandas as pd
"""
Gallery 使用 pyecharts 1.1.0
参考地址:  https://echarts.apache.org/examples/editor.html?c=area-rainfall

目前无法实现的功能:

1、dataZoom 放大的时候左侧 Y 轴无法锁定上下限 (待解决, 配置无关)
2、X 轴的日期格式化的值在 Python 中无法良好的渲染到 js 中
"""
data1 = pd.read_csv(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最大值.csv")
#print(data1)
x = data1["Time"]
x_data1 = [d.replace(" ", "-") for d in x]
y_max = data1["普通工业"]
print(y_max)
data2 = pd.read_csv(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最小值.csv")
y_min = data2["普通工业"]
(
    Line(init_opts=opts.InitOpts(width="1680px", height="800px"))
    .add_xaxis(xaxis_data=x_data1)
    .add_yaxis(
        series_name="有功功率最大值",
        y_axis=y_max,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
        linestyle_opts=opts.LineStyleOpts(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="有功功率最小值",
        y_axis=y_min,
        yaxis_index=1,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
        linestyle_opts=opts.LineStyleOpts(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            name="有功功率最小值",
            name_location="start",
            type_="value",
            #max_=5,
            is_inverse=True,
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        )
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="普通工业有功功率突变图",
            #subtitle="数据来自西安兰特水电测控技术有限公司",
            pos_left="center",
            pos_top="top",
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_left="left"),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
        ],
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(name="有功功率最大值（kw）", type_="value"),
    )

    .render("rainfall.html")
)
