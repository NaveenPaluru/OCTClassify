#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:15:31 2020

@author: cds
"""


from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
from config import Config
from mydataset import OCTTest
import h5py as h5
from MiniCNN import MiniCNN
import torch.nn.functional as F
import sklearn.metrics as metrics
import seaborn as sn
import pandas  as pd
import scipy.io


def test(directory):    
    net = MiniCNN()
    net.load_state_dict(torch.load(directory))    
    config  = Config()
    # Download the data and place it in ./UCSD Data/Data/        
    # make the data iterator for testing data . 
    test_data = OCTTest('./F2test.csv','./UCSD Data/Data/')
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False, num_workers=2)           
    if config.gpu == True:
        net.cuda(config.gpuid).eval()    
    for i,data in tqdm.tqdm(enumerate(testloader)):             
        # start iterations
        images,imtruth = Variable(data[0]),Variable(data[1])
        # ckeck if gpu is available
        if config.gpu == True:
            images  = images.cuda(config.gpuid)
            imtruth = imtruth.cuda(config.gpuid)
        # make forward pass      
        output = F.softmax(net(images),dim=1)
        _, pred= torch.max(output, dim=1)  
        if i==0:
            tmp = pred.cpu().detach()
            tmpl=imtruth.cpu().detach()
        else:
            tmp = torch.cat((tmp ,pred.cpu().detach()),dim=0)
            tmpl= torch.cat((tmpl,imtruth.cpu().detach()),dim=0)     
            
    return tmp,tmpl.squeeze()
                
            
           
if __name__ == '__main__':         
        
    saveDir='./'
        
    # if want to test on a specific model
    directory=saveDir+"MiniCNN_50_model.pth"
    print('Loading the Model : ', directory)
    
    tmp, tmpl = test(directory)
    tmp  = tmp.numpy().astype(np.float64 )  
    tmpl = tmpl.numpy().astype(np.float64)      
    hh=np.reshape(tmp, (-1,1))      
    gg=np.reshape(tmpl,(-1,1))        
    matrix = metrics.confusion_matrix(gg,  hh, normalize='true')   
    df_cm = pd.DataFrame(matrix, index  = ['CNE', 'DME', 'DRU', 'NRM'],
                                columns = ['CNE', 'DME', 'DRU', 'NRM'])
    #plt.figure(figsize = (10,10))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt = '.2f') # font sizesn.set(font_scale=1.4) # for label size
    plt.tight_layout()
    plt.ylabel('Target Class')    
    plt.xlabel('Output Class') 
    plt.show()
    
    print(classification_report(gg,hh))
    
       
    




