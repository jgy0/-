import pandas as pd

df = pd.read_excel('./data/附件4.xlsx')

data = pd.DataFrame(columns=['序号', '原料名称', '百分比（%）'])
# print(data)

for i in range(len(df)):
    yuanliao = []
    zhanbi = []
    st = df['原料与占比'][i]
    xuhao = df['序号'][i]
    stf = list(st.split(","))

    for co in stf:
        # print(co)
        l1 = list(co.split(" (占"))
        if len(l1) == 2 :
            yuanliao.append(l1[0])
            zhanbi.append(l1[1][:-1])
    for k in range(len(yuanliao)):
        data = data.append({'序号': xuhao, '原料名称': yuanliao[k], '百分比（%）': zhanbi[k]}, ignore_index=True)
        # print(data)
    # print(i)

# print(data)
data.to_excel('./result/result4_2.xlsx', index = False)