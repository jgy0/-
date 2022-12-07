import pandas as pd

df = pd.read_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\附件1.xlsx', sheet_name='安徽省')
df['含磷百分比'] = df['P2O5百分比'].apply(lambda x: x*(62/152))
df['含钾百分比'] = df['K2O百分比'].apply(lambda x: x*(78/94))
df['总无机养分百分比'] = df['含磷百分比']+df['含钾百分比']+df['总氮百分比']
df['总无机养分百分比'] = df['总无机养分百分比'].apply(lambda x: round(x, 3))
# df['总无机养分百分比'].apply(lambda x: '%.3f%%' % (x*100))
df = df[['序号', '正式登记证号', '总无机养分百分比']]
df.to_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\result\result1_2.xlsx', index=False)
