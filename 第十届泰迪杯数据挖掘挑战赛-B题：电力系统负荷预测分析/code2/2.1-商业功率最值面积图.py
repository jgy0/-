import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker
import pandas as pd
data1 = pd.read_csv(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最大值.csv")
#print(data1)
x = data1["Time"]
x_data1 = [d.replace(" ", "-") for d in x]
y_max = data1["商业"]
print(y_max)
data2 = pd.read_csv(r"C:\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最小值.csv")
y_min = data2["商业"]
c = (
    Line()
    .add_xaxis(x_data1)
    .add_yaxis("有功功率最大值", y_max, areastyle_opts=opts.AreaStyleOpts(opacity=0.5))
    .add_yaxis("有功功率最小值", y_min, areastyle_opts=opts.AreaStyleOpts(opacity=0.5))
    .set_global_opts(title_opts=opts.TitleOpts(title="商业功率最值面积图"))
    .render("line_area_style.html")
)
