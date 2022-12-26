# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset,DataLoader

class HumanDataset(Dataset):
    def __init__(self,df):
        super(HumanDataset, self).__init__()
        self.image=df.copy()
        self.image_id=self.image['Id'].values             # 不加.value会出现问题
        target=np.zeros((len(df),28))
        for i,label in enumerate(df['Target']):
            label=[int(x) for x in label.split()]
            for l in label:
                target[i,l]=1
        self.target=target
        pass

    def __len__(self):
        x=len(self.image)
        return len(self.image)

    def __getitem__(self, item):
        img_path=self.image_id[item]
        r = np.array(Image.open(img_path + "_red" + '.png'))
        g = np.array(Image.open(img_path + "_green" + '.png'))
        b = np.array(Image.open(img_path + "_blue" + '.png'))

        images = np.zeros(shape=(512, 512, 3))
        images[:, :, 0] = r.astype(np.uint8)
        images[:, :, 1] = g.astype(np.uint8)
        images[:, :, 2] = b.astype(np.uint8)

        X = np.transpose(images, axes=(2, 0, 1))
        X = X.astype(np.float32)
        y=self.target[item]
        return torch.FloatTensor(X), self.target[item]


def predict(model,data_loader):
    with torch.no_grad():  # evaulate
        model.eval()
        pred = []
        y = []
        for _, (images, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                images = images.cuda() # non_blocking=True
            output = model(images)
            pred += output.cpu().data.tolist()
            y += target.tolist()
    return y,pred


class MyModel(nn.Module):
    def __init__(self,inplanes,planes):
        super(MyModel, self).__init__()
        pass

