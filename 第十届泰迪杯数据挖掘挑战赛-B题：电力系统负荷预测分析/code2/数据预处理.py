import pandas as pd
#data = pd.read_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\附件2-行业日负荷数据副本.csv",encoding = 'gbk')
import pandas as pd
def missing_value_processing():
    data = pd.read_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\附件2-行业日负荷数据副本.csv",encoding = 'gbk')#读数据文件
    print(data.columns)#获取数据列名,即获取表头
    print("原始数据量："+str(len(data)))
    #print("\n")
    print("缺失值处理starting...")
    print('缺失值数量：')#显示缺失值数量
    print(data.isnull().sum())

    data = data.fillna(data.interpolate())

    #data.to_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\附件2-行业日负荷数据经处理.csv")

    '''
    常用填充值：df.fillna(0) # 填充为0
              df.fillna(df.mean())#填充为均值
              df.fillna(df.median())#填充为中位数
              df.fillna(df.mode())#填充为众数
              df.fillna(method='pad') # 填充前一条数据的值
              df.fillna(method='bfill') # 填充后一条数据的值
    '''
    print('经缺失值处理后的数据量：' + str(len(data)))
    print('缺失值处理后的结果！')
    print(data.isnull().sum())
#---3.不处理
    print('缺失值处理完成！！！')
    print("\n")

#--------------------异常值处理（删除/填充/不处理）-----------------
'''
def Outliers_processing():
    data = pd.read_csv('../..', encoding ='gbk')
    print("原始数据量：" + str(len(data)))
    print("\n")
    print("异常值处理starting...")
    print('异常值统计：')
    print(data[data.某一列列名 == '--'].shape)#把--替换成异常值
    #可以模仿这个，先写一个填充函数：
    def fill(x):
        if x.recently_logged == '--':
            x.recently_logged = x.register_time
        return x
    data = data.apply(fill, axis='columns')
    print('经处理后异常值统计：')
    print(data[data.某一列列名 == '--'].shape)  # 把--替换成异常值
    print('经异常值处理后数据量：' + str(len(data)))
    print("\n")
    print('异常值处理完成！！！')
    print("\n")
'''
#----------------------重复值处理-----------------------------
def duplicate_value_processing():
    data = pd.read_csv("C:\\Users\Lenovo\Desktop\泰迪杯\全部数据\附件2-行业日负荷数据副本.csv",encoding = 'gbk')
    print("原始数据量：" + str(len(data)))
    print("\n")
    print("重复值处理starting...")
#1.去除完全重复的行数据
    data.drop_duplicates(inplace=True)
#2.去除指定列重复的行数据
    #data.drop_duplicates(subset=[], keep='', inplace=True)
    print('经重复值处理后数据量：' + str(len(data)))
    print('重复值数量：0')
    print('重复值处理完成！！！')
    print("\n")

if __name__ == '__main__':
    missing_value_processing()
    #Outliers_processing()
    duplicate_value_processing()


