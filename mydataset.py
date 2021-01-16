#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:54:31 2020

@author: cds
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
from skimage.transform import resize
import numpy as np
import torchvision.transforms as transforms
from skimage.color import rgb2gray


class OCTTrain(Dataset):
    def __init__(self, csv_file, root_dir, transform =  transforms.Compose([ 
                 transforms.ToTensor()]) , Flag=True):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.Flag = Flag
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):   
        if self.Flag:
            img_name = os.path.join(self.root_dir,
                                    self.names['Name'][idx])
        else:
             img_name = self.names['Name'][idx]   
        image = io.imread(img_name)
        if len(image.shape)==3:
            image = rgb2gray(image)
        image = resize(image, (224, 224))
        label = self.names['Label'][idx]
        if self.transform:
            image = self.transform(image)
            image = image.float()
            label = torch.tensor(label)
        return image,label      

    
class OCTTest(Dataset):
    def __init__(self, csv_file, root_dir, transform =  transforms.Compose([ 
                 transforms.ToTensor()]) , Flag=True):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.Flag = Flag
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):   
        if self.Flag:
            img_name = os.path.join(self.root_dir,
                                    self.names['Name'][idx])
        else:
             img_name = self.names['Name'][idx]   
        image = io.imread(img_name)
        if len(image.shape)==3:
            image = rgb2gray(image)
        image = resize(image, (224, 224))
        label = self.names['Label'][idx]
        if self.transform:
            image = self.transform(image)
            image = image.float()
            label = torch.tensor(label)
        return image,label   