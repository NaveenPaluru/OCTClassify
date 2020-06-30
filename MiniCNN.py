#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:59:07 2020

@author: cds
"""


import torch.nn as nn
import torch

class Bottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding =1,
                 dilation=1,
                 dropout_prob=0.1,
                 assymetric = False,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - shortcut connection

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        if assymetric:
            self.ext_conv2 = nn.Sequential(
                    nn.Conv2d(
                        internal_channels,
                        internal_channels,
                        kernel_size=(kernel_size, 1),
                        stride=1,
                        padding=(padding, 0),
                        dilation=dilation,
                        bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                    nn.Conv2d(
                        internal_channels,
                        internal_channels,
                        kernel_size=(1, kernel_size),
                        stride=1,
                        padding=(0, padding),
                        dilation=dilation,
                        bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        # Add main and extension branches
        out = main + ext
        return self.out_activation(out)

class MiniCNN(nn.Module): 
    def __init__(self):
          super(MiniCNN, self).__init__()
          # Setting up the Sequential of CNN Layers
          self.cnn = nn.Sequential( 
              
          nn.Conv2d(1, 64,    kernel_size=3,stride=2,padding=0),
          nn.LeakyReLU(0.05),
          
          nn.Conv2d(64, 128,  kernel_size=3,stride=2,padding=0),
          nn.LeakyReLU(0.05),
          
          
          Bottleneck(128),
          Bottleneck(128 , assymetric = True),  
          
                  
          nn.Conv2d(128, 256,  kernel_size=3,stride=2,padding=0),
          nn.LeakyReLU(0.05),
          
          
          Bottleneck(256),
          Bottleneck(256 , assymetric = True),
          
                      
          nn.Conv2d(256, 128,   kernel_size=3,stride=2,padding=0),
          nn.LeakyReLU(0.05),
          
          Bottleneck(128),
          Bottleneck(128 , assymetric = True),    
          
               
          
          nn.Conv2d(128, 16,   kernel_size=3,stride=2, padding=1),
          nn.LeakyReLU(0.05),          
          )    
          
          self.fc = nn.Linear(784,4)
          
    def forward(self,x):
        x = self.cnn(x)
        x = self.fc(x.view(x.size()[0], -1))
        return x
         
    
    def getnumberofparams(self,model):
        pp=0
        for p in (model.parameters()):
            nn=1
            for s in (p.size()):
                nn = nn*s
            pp += nn
        return pp

    
if __name__=="__main__" :
    net = MiniCNN()
    net = net.cuda(0)
    pp = net.getnumberofparams(net)
    print('Parameters : ', pp)
    x = torch.rand(1,1,224,224).cuda(0)
    import time
    t0= time.clock()
    y = net(x)
    t1 = time.clock() - t0    
    print('Elapsed Time in sec is ', t1 )    
    print('Output Size :' , y.size())
    size = (pp * 32) / (8*1024*1024)
    print('Model Size (MB) :{:.2f}'.format(size))
    