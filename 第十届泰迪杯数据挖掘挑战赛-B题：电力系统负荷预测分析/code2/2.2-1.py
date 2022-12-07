import pandas as pd
#----------------------------------------------1.首先根据附件2按照时间将四个行业组织起来得到功率最大值和最小值csv
df = pd.read_csv(r'C:\Users\Lenovo\Desktop\泰迪杯\全部数据\附件2-行业日负荷数据经处理.csv')
metrics_df1=pd.pivot_table(df,values='有功功率最大值',index='数据时间',columns='行业类型')
metrics_df1.reset_index(inplace=True)
metrics_df1.fillna(0,inplace=True)
metrics_df1 = metrics_df1.rename(columns={'数据时间':'Time'})
print(metrics_df1)
data2 = metrics_df1.to_csv('C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最大值.csv')

metrics_df2=pd.pivot_table(df,values='有功功率最小值',index='数据时间',columns='行业类型')
metrics_df2.reset_index(inplace=True)
metrics_df2.fillna(0,inplace=True)
metrics_df2 = metrics_df2.rename(columns={'数据时间':'Time'})
print(metrics_df2)
data2 = metrics_df2.to_csv('C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\有功功率最小值.csv')