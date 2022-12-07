# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import time
import copy
import random
import sys
import shutil
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import transforms,models
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
"""
读入标签
"""
df=pd.read_csv('../data/message.txt',encoding='gbk',header=None,names=[1,2])
print(df.head(5).append(df.tail(5)))
df.info()
df=df.dropna()

df[1]=df[1].apply(lambda x:x.replace("['",""))
df[2]=df[2].apply(lambda x:x.replace("']","").replace(' ',""))

# df['length']=df[2].apply(lambda x:len(x))   # 都是2说明有空格符号

df[2]=df[2].apply(lambda x:ord(x))
print(df.head(5).append(df.tail(5)))
df.info()
df[2]=df[2].apply(lambda x:x-48 if (x>=48 and x<=57) else x)
df[2]=df[2].apply(lambda x:x-55 if (x>=65 and x<=90) else x)
df[2]=df[2].apply(lambda x:x-61 if (x>=97 and x<=122) else x)
print(df)
# df.to_csv(r'../data/label1.csv',index=None)

"""
训练模型
"""
# 查看一下图片的形状
img1=Image.open('../data/image_message/im10001.jpg')
img1.size
img1=np.array(img1)
print(img1.shape)  # (400, 400, 3)
# plt.imshow(img1)
# plt.show()
"""
划分训练集和测试集
"""
# split_train_test




# 图片归一化标准化
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Resize((400,400)),
                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                              ])


class Mydata(Dataset):
    def __init__(self,path,label_path):
        self.path=path
        self.path_list=os.listdir(self.path)
        self.path_label=label_path
        if self.path_label is not None:
            df = pd.read_csv(self.path_label)
            self.y = list(df.iloc[:, 1])


    def __getitem__(self, item):
        names=self.path_list[item]
        img_name=os.path.join(self.path, names)
        img=Image.open(img_name)
        # img = img.convert("RGB")
        # label
        if self.path_label is not None:
            return transform(img), torch.tensor(self.y[item])
        else:
            return transform(img)

    def __len__(self):
        return len(self.path_list)

def im_convert(tensor):
    """展示"""
    image=tensor.to("cpu").clone().detach()
    image=image.numpy().squeeze()
    image=image.transpose(1,2,0)
    # image=image.clip(0,1)   # numpy.clip(a, a_min, a_max, out=None) 将数组中的元素限制在a_min, a_max之间

    return image


train_path='../data/image_message'
valid_path='../data/image_message_val'
train_label='../data/train.csv'
valid_label='../data/val.csv'
batch_size=50

train_data=Mydata(train_path,train_label)
valid_data=Mydata(valid_path,valid_label)


# print(train_data[0])
# print(train_data[1])
# print(type(train_data[0]))
# print(train_data[0][0].size())

train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=True)
valid_loader=DataLoader(dataset=valid_data,batch_size=batch_size,shuffle=True,drop_last=True)


fig=plt.figure(figsize=(20,12))
columns=4
rows=2
dataiter=iter(train_loader)
inputs, classes=dataiter.next()

for idx in range(columns*rows):
    ax=fig.add_subplot(rows,columns,idx+1)
    n = classes[idx]   #  n:tensor(62)
    ax.set_title(n)
    plt.imshow(im_convert(inputs[idx]))
plt.show()

data_loaders={'train':train_loader,'valid':valid_loader}


model_name='resnet'
feature_extract=True   # 是否用别人训练好的
train_on_gpu=torch.cuda.is_available()
if not train_on_gpu:
    print("GPU is not available, train on CPU")
else:
    print("GPU is available, train on GPU")

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model,feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad=False

# model_ft=models.resnet152()
# print(model_ft)

def initialize_model(model_name,num_classes,feature_extract,use_pretrained=True):
    if model_name=='resnet':
        model_ft=models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)   # 是否冻住前面
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                   nn.LogSoftmax(dim=1))

    return model_ft

model_ft=initialize_model('resnet',62,feature_extract)  # 几分类
model_ft=model_ft.to(device)
filename='checkpoint.pth'
para_learn=model_ft.parameters()
if feature_extract:
    para_learn=[]
    for name,parameter in model_ft.named_parameters():
        if parameter.requires_grad==True:
            para_learn.append(parameter)
            print('\t', name)
else:
    for name,pare in model_ft.named_parameters():
        if pare.requires_grad==True:
            print('\t',name)

optimizer_ft=optim.Adam(para_learn,lr=1e-2)
scheduler=optim.lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)
criterion=nn.NLLLoss()

def train_model(models,dataloader,criterion,optimizer,num_epochs=25,filename=filename):
    start=time.time()
    best_acc=0
    best_model_wts=copy.deepcopy(models.state_dict())
    train_loss=[]
    valid_loss=[]
    train_acc=[]
    valid_acc=[]
    models=models.to(device)
    LRs = [optimizer.param_groups[0]['lr']]

    for epoch in range(num_epochs):
        print('{}/{}'.format(epoch+1,num_epochs))
        for phase in ['train','valid']:
            if phase =='train':
                models.train()
                print('this is train process')
            else:
                models.eval()
                print('this is valid process')
            running_loss = 0
            running_acc = 0

            for inputs, label in dataloader[phase]:
                inputs = inputs.to(device)
                # [8, 3, 240, 108])
                label =label.to(device)
                outputs=models(inputs)
                optimizer.zero_grad()
                loss=criterion(outputs,label)
                _,pred=torch.max(outputs,1)
                if phase =='train':
                    loss.backward()
                    optimizer.step()


                running_loss+=loss.item()
                running_acc+=torch.sum(pred==label.data)
            epoch_loss=running_loss/num_epochs
            epoch_acc=running_acc.double()/len(dataloader[phase].dataset)
            time_cost=time.time()-start
            print('Time cost ',time_cost//60,'min',time_cost % 60,'s')
            print('{} loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
# this

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(models.state_dict())
                state = {
                    'state_dict': models.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                valid_acc.append(epoch_acc)
                valid_loss.append(epoch_loss)
                scheduler.step()
            if phase == 'train':
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()


    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    models.load_state_dict(best_model_wts)
    return models, valid_acc, train_acc, valid_loss, train_loss, LRs

# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(
#     model_ft, data_loaders, criterion, optimizer_ft, num_epochs=3)

"""
加载模型
"""
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
"""
预测数据
"""
pred=[]
test_path='../data/test_images'
test_label=None
test_data=Mydata(test_path,test_label)
train_loader=DataLoader(dataset=test_data,batch_size=batch_size,drop_last=False)

for dataiter in iter(train_loader):
    images = dataiter

    model_ft.eval()

    if train_on_gpu:
        output = model_ft(images.cuda())
    else:
        output = model_ft(images)

    _, preds_tensor = torch.max(output, 1)

    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    pred.extend(preds)


test_names=os.listdir('../data/test_images')
test_label=pd.DataFrame(test_names)
test_label['1']=pred

test_label.to_csv('submit.csv',index=None,header=None)
# 得到一个batch的测试数据
