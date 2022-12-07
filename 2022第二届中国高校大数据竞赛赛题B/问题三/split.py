# -*- coding:utf-8 -*-
import os
import random
import shutil

val_percent=0.1
source_path_mess="../data/train/image_message"
source_path_origin="../data/train/image_original"

des_path_mess="../data/valid/image_message_val"
des_path_origin="../data/valid/image_original_val"

# if not os.path.exists(des_path_mess):
#     os.makedirs(des_path_mess)
#     print("创建{}文件夹".format(des_path_mess))
# if not os.path.exists(des_path_origin):
#     os.makedirs(des_path_origin)
#     print("创建"+des_path_origin+"文件夹")


def split_data(sour_path,des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)
        print("创建{}文件夹".format(des_path))
    files1=os.listdir(sour_path)
    num1=range(len(files1))
    select_num1=int(val_percent*len(files1))
    file_select=random.sample(num1,select_num1)
    for i in file_select:
        name=files1[i]
        file_name=sour_path+'/'+name
        des_file_name=des_path+'/'+name
        shutil.move(file_name,des_file_name)


if __name__=="__main__":
    split_data(source_path_mess,des_path_mess)
    split_data(source_path_origin,des_path_origin)




