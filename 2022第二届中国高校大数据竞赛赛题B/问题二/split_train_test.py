# -*- coding:utf-8 -*-
import os
import random
import re
import sys
import shutil
import pandas as pd
# df=pd.read_csv('../data/label1.csv')
#
# source_path="../data/image_message"
# de_path="../data/image_message_val"
# if not os.path.exists(de_path):
#     os.makedirs(de_path)
#     print('创建'+de_path)
# val_percent=0.1
#
# filename=os.listdir(source_path)
# num=len(filename)
# val=int(num*val_percent)
# file=range(num)
# print(file)
# file_select=random.sample(file,val)
# df_val=df.iloc[file_select,:]
# df_val.to_csv("../data/val.csv",index=None)
# df_train=df.copy()
# df_train.drop(file_select,inplace=True)
# df_train.to_csv("../data/train.csv",index=None)
# print("运行到行号：",sys._getframe().f_lineno)
# for i in file_select:
#
#     val_name=filename[i]
#     file_name_source=source_path+'/'+val_name   # 源文件路径
#     val_path_new=de_path+'/'+val_name          # 目标文件路径
#     shutil.move(file_name_source,val_path_new)
# print("结束 运行到行号：",sys._getframe().f_lineno)
"""
val.csv 排序
"""
data=pd.read_csv('../data/val.csv')
data[3]=data.iloc[:,0].apply(lambda x:re.findall('\d+',x)[0])
data=data.sort_values(by=3)
data=data.drop(3,axis=1) # 列
print(data)
data.to_csv("../data/val.csv",index=None)
# re.findall('\d+', 'im10603') # ['10603']
