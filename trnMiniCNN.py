#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:20:52 2020

@author: naveen_p
"""


import os,time
import sklearn.metrics as metrics
import scipy.io
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
import torch.nn as nn
from datetime import datetime
from config import Config
from mydataset import OCTTrain
from MiniCNN import MiniCNN
import torch.optim as optim
from torch.optim import lr_scheduler


print ('*******************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+'model'
print('Model will be saved to  :', directory)
if not os.path.exists(directory):
     os.makedirs(directory)
config  = Config()
# Download the data from the link given in read me file and place it in ./UCSD Data/Data/
# make the data iterator for training data
train_data = OCTTrain('./F2train.csv','./UCSD Data/Data/')
trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batchsize, shuffle=True, num_workers=2)




print('----------------------------------------------------------')
#%%
# Create the object for the network
if config.gpu == True:    
    net = MiniCNN()
    net.cuda(config.gpuid)
    net.train()       
else:
   net = MiniCNN()
   
# Define the optimizer
optimizer = optim.Adam(net.parameters(),lr=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Define the loss function
criterion = nn.CrossEntropyLoss()


# Iterate over the training dataset
train_loss = []

for j in range(config.epochs):  
    # Start epochs   
    runtrainloss = 0
    for i,data in tqdm.tqdm(enumerate(trainloader)): 
        # start iterations
        images,trainLabels = Variable(data[0]),Variable(data[1])        
        # ckeck if gpu is available
        if config.gpu == True:
            images  = images.cuda(config.gpuid )
            trainLabels = trainLabels.cuda(config.gpuid)                    
        # make forward pass      
        output = net(images)       
        #compute loss
        loss   = criterion(output, trainLabels)                
        # make gradients zero
        optimizer.zero_grad()        
        # back propagate
        loss.backward()        
        # Accumulate loss for current minibatch
        runtrainloss += loss.item()       
        # update the parameters
        optimizer.step()      
    # print loss after every epoch    
    print('\n Training - Epoch {}/{}, loss:{:.4f} '.format(j+1, config.epochs, runtrainloss/len(trainloader)))
    train_loss.append(runtrainloss/len(trainloader))       
    # Take a step for scheduler
    scheduler.step()
    print('----------------------------------------------------------')    
    #save the model   
    torch.save(net.state_dict(),os.path.join(directory,"MiniCNN_" + str(j+1) +"_model.pth"))    	    

# Save the train stats

np.save(directory + '/trnloss.npy',np.array(train_loss) )

# plot the training loss

x = range(config.epochs)
plt.figure()
plt.plot(x,train_loss,label='Training')
plt.xlabel('epochs')
plt.ylabel('Train Loss ') 
plt.legend(loc="upper left")  
plt.show()
  


