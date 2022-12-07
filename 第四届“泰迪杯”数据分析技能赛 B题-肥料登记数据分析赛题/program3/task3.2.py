import pandas as pd

dfa = pd.read_excel('./result/result2_2.xlsx')

dfy = dfa[dfa['有效期'] > '2021-09']
dfy = dfy[dfy['有效期'] != '2021-2']
dfy = dfy[dfy['有效期'] != '2021-09']
dfy.to_excel('./result/result3_2.xlsx', index = False)


df_e = dfy[dfy['正式登记证号'].str.contains('鄂')]
df_gui = dfy[dfy['正式登记证号'].str.contains('桂')]
# print(df_gui)

ezu = list(df_e['分组标签'])
guizu = list(df_gui['分组标签'])

def counter(list):
    from collections import Counter
    # 对列表内的元素进行排序，计数，以字典的形式返回
    list_counter = Counter(list)
    list_sorted = sorted(list_counter.items(), key = lambda x:x[1], reverse = True)
    list_dict = dict(list_sorted)
    return list_dict

de = counter(ezu)
dgui = counter(guizu)
print(de)
print(dgui)