import pandas as pd

df = pd.read_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\附件1.xlsx', sheet_name='安徽省')
df.loc[df['产品通用名称'] == '掺混肥料', '产品通用名称'] = '复混肥料'
df.loc[df['产品通用名称'] == '稻苗床土调酸剂', '产品通用名称'] = '床土调酸剂'
df['产品通用名称'] = df['产品通用名称'].apply(lambda x: x.replace('－', '-'))
df['产品通用名称'] = df['产品通用名称'].apply(lambda x: " ".join(x.split()))
df['产品通用名称'] = df['产品通用名称'].apply(lambda x: x.replace(' ', ''))
# " ".join(s.split())
# df['产品通用名称'] = df['产品通用名称'].replace(' ', '')
df.loc[df['产品通用名称'] == '有机无机复混肥料', '产品通用名称'] = '有机-无机复混肥料'
df.loc[df['产品通用名称'] == '掺混肥料', '产品通用名称'] = '复混肥料'

df.to_excel(r'D:\xinjianqq\比赛练习\B题-肥料登记数据分析赛题\result\result1_1.xlsx', index=False)
