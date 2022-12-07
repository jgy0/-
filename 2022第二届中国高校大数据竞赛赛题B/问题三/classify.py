# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-

import copy
import json
import time

import pandas as pd
from torch.utils.data import Dataset,DataLoader,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.functional as F
from PIL import Image,ImageFile
from torchvision import transforms,datasets,models
import os
from torch import nn
import torch.optim as optim
ImageFile.LOAD_TRUNCATED_IMAGES = True
print(torch.__version__)

data_dir='../data'
train_dir=data_dir+'/train'
valid_dir=data_dir+'/valid'

data_transforms={
    'train':transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize([400, 400]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),
        transforms.RandomGrayscale(p=0.025),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'valid':transforms.Compose([
        transforms.Resize([400, 400]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
}


batch_size=8
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
data_loaders={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True) for x in ['train','valid']}
data_size={x:len(image_datasets[x]) for x in ['train','valid']}
class_names=image_datasets['train'].classes
print(image_datasets['train'].classes)
print(image_datasets['train'].class_to_idx)

image_datasets


def im_convert(tensor):
    """展示"""
    image=tensor.to("cpu").clone().detach()
    image=image.numpy().squeeze()
    image=image.transpose(1,2,0)
    image=image*np.array((0.229, 0.224, 0.225))+np.array((0.485, 0.456, 0.406))
    image=image.clip(0,1)   # numpy.clip(a, a_min, a_max, out=None) 将数组中的元素限制在a_min, a_max之间

    return image


fig=plt.figure(figsize=(20,12))
columns=4
rows=2
dataiter=iter(data_loaders['valid'])
inputs, classes=dataiter.next()

for idx in range(columns*rows):
    ax=fig.add_subplot(rows,columns,idx+1)
    n = classes[idx]   #  n:tensor(62)
    ax.set_title(int(classes[idx]))
    plt.imshow(im_convert(inputs[idx]))
plt.show()

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

model_ft=models.resnet152()
model_ft
# for name,parameter in  model_ft.named_parameters():
#     print(name,parameter)
#
def initialize_model(model_name,num_classes,feature_extract,use_pretrained=True):
    if model_name=='resnet':
        model_ft=models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)   # 是否冻住前面
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                   nn.LogSoftmax(dim=1))
        input_size = 224

    return model_ft,input_size

model_ft,input_size=initialize_model('resnet',2,feature_extract)
model_ft=model_ft.to(device)
filename='checkpoint3.pth'
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
            else:
                models.eval()

            running_loss = 0
            running_acc = 0

            for inputs, label in dataloader[phase]:
                inputs = inputs.to(device)
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


# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
"""
验证集
"""
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Resize((400,400)),
                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                              ])

class Mydata3(Dataset):
    def __init__(self,path):
        self.path=path
        self.file_list=os.listdir(self.path)
    def __getitem__(self, item):
        names = self.file_list[item]
        img_name = os.path.join(self.path, names)
        img = Image.open(img_name)
        return transform(img)
    def __len__(self):
        return len(self.file_list)


test_data=Mydata3("../data/test_images")
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,drop_last=False)
pred=[]
for dataiter in iter(test_loader):
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

test_label.to_csv('submit3.csv',index=None,header=None)


# def process_image(image_path):
#     # 读取测试数据
#     img = Image.open(image_path)
#     # 相同的预处理方法
#     mean = np.array([0.485, 0.456, 0.406])  # provided mean
#     std = np.array([0.229, 0.224, 0.225])  # provided std
#     img = (img - mean) / std
#
#     # 注意颜色通道应该放在第一个位置
#     img = img.transpose((2, 0, 1))
#     return img
#
#
# def imshow(image, ax=None, title=None):
#
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # 颜色通道还原
#     image = np.array(image).transpose((1, 2, 0))
#
#     # 预处理还原
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
#     image = np.clip(image, 0, 1)
#
#     ax.imshow(image)
#     ax.set_title(title)
#
#     return ax
#
# image_path = '../data/test_images/im19951.jpg'
# img = process_image(image_path)
# imshow(img)

# 得到一个batch的测试数据
dataiter = iter(data_loaders['valid'])
images, labels = dataiter.next()

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
preds


fig=plt.figure(figsize=(20, 20))
columns =4
rows = 2

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(str(preds[idx]), str(labels[idx].item())),
                 color=("green" if str(preds[idx])==str(labels[idx].item()) else "red"))
plt.show()

