import torch
from torch import nn
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import  Image
import pathlib
import argparse
from cust_dataset import cifarDS
from utils.ImageTransform import ImageTransform
from utils.dataDownload import DownloadAndExtract
from engine import AlexNet

from train import train_script


if os.path.exists(os.path.join(os.getcwd(),'cifar10')):
    print('cifar10 exist already')
else:
    print('no such file, downloading')
    DownloadAndExtract()

train_dir = os.path.join(os.getcwd(),'cifar10','train')
test_dir  = os.path.join(os.getcwd(),'cifar10','test')



device = "cuda" if torch.cuda.is_available() else 'cpu'
device

trainDataset = cifarDS(train_dir,ImageTransform)
testDataset = cifarDS(test_dir,ImageTransform)

trainDataLoader = DataLoader(trainDataset,  batch_size=2,shuffle=True)
testDataLoader  = DataLoader(testDataset,batch_size =1,shuffle=False)

alexnet = AlexNet()
alexnet.to(device)


optim = torch.optim.Adam(alexnet.parameters(),)
criteration = nn.CrossEntropyLoss()

train_script(alexnet,optim,criteration,trainDataLoader,testDataLoader,device,1000)




