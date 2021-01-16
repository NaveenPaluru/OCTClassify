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

# if not os.path.exists(directory):
#     os.makedirs(directory)

config  = Config()

weights = np.array([1.38, 4.27, 5.87, 1.00])


# make the data iterator for training data
train_data = OCTTrain('./UCSD Data/AuthorFold/F1train.csv','./UCSD Data/AuthorFold/Data/')
trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batchsize, shuffle=True, num_workers=2)




print('----------------------------------------------------------')
#%%
# Create the object for the network

if config.gpu == True:    
    net = MiniCNN()
    net.cuda(config.gpuid)
    net.train() 
    #class_weights = torch.FloatTensor(weights).cuda(config.gpuid)
    
else:
   net = MiniCNN()
   
# Define the optimizer
optimizer = optim.Adam(net.parameters(),lr=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Define the loss function
criterion = nn.CrossEntropyLoss()


# Iterate over the training dataset
train_loss = []
# val_loss = []
# acc = []


for j in range(config.epochs):  
    # Start epochs   
    runtrainloss = 0
    #runvalloss = 0
    
    
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
    
    # net.eval()
    # total =0
    # correct = 0
    # for i,data in tqdm.tqdm(enumerate(valloader)): 
    #     # start iterations
    #     images,valLabels = Variable(data[0]),Variable(data[1])
        
    #     # ckeck if gpu is available
    #     if config.gpu == True:
    #         images  = images.cuda(config.gpuid )
    #         valLabels = valLabels.cuda(config.gpuid)
                    
    #     # make forward pass      
    #     output = net(images)
        
    #     # Find Accuracy
    #     _, predicted = torch.max(F.softmax(output, dim=1), dim=1)
    #     total += valLabels.size(0)
    #     correct += (predicted == valLabels).sum().item()
       
    #     #compute loss
    #     loss   = criterion(output, valLabels)        
                
    #     # Accumulate loss for current minibatch
    #     runvalloss += loss.item()
        
           
    # # print loss after every epoch
    
    # print(' \n Validatn - Epoch {}/{}, loss:{:.4f}, Acc:{:.4f}'.format(j+1, 
    #             config.epochs, runvalloss/len(valloader), correct / total))
    # val_loss.append(runvalloss/len(valloader))
    # acc.append(correct / total)
    
    print('----------------------------------------------------------')
    
    
    #save the model   
    torch.save(net.state_dict(),os.path.join(directory,"MiniCNN_" + str(j+1) +"_model.pth"))
    
    
    	    

# Save the train stats

np.save(directory + '/trnloss.npy',np.array(train_loss) )
# np.save(directory + '/valloss.npy',np.array(val_loss) )
# np.save(directory + '/acc.npy',    np.array(   acc    ) )


# plot the training loss

x = range(config.epochs)
plt.figure()
plt.plot(x,train_loss,label='Training')
plt.xlabel('epochs')
plt.ylabel('Train Loss ') 
plt.legend(loc="upper left")  
plt.show()
# plt.figure()
# plt.plot(x,val_loss,label='Validation')
# plt.xlabel('epochs')
# plt.ylabel('Val Loss ') 
# plt.legend(loc="upper left")  
# plt.show()
# plt.figure()
# plt.plot(x,acc,label='Validation')
# plt.xlabel('epochs')
# plt.ylabel('Val acc ') 
# plt.legend(loc="upper left")  
# plt.show()
                      


