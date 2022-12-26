# -*- coding:utf-8 -*-
from torchvision.models import *
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.optim as optim
import pretrainedmodels
from utils import *
import torch
import torch.functional as F
import torchvision
import matplotlib.pyplot as plt

def xception_model(cfg):
    model = pretrainedmodels.models.xception(pretrained='imagenet')
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    model.last_linear = nn.Sequential(
        nn.Linear(2048, 28),
        nn.Sigmoid()
    )

    return model

