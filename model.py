import numpy as np
import random
import os 
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast

from torchsummary import summary
from sklearn.metrics import r2_score

from ray import tune

import json
import itertools
from itertools import groupby
import gzip 
from io import BytesIO
from time import time 

import matplotlib.pyplot as plt

import pyBigWig
from scipy.sparse import csc_matrix
import math 

class upd_GELU(nn.Module):
    def __init__(self):
        super(upd_GELU, self).__init__()
        self.constant_param = nn.Parameter(torch.Tensor([1.702]))
        self.sig = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        outval = torch.mul(self.sig(torch.mul(self.constant_param, input)), input)
        return outval
    
def ones_(tensor: Tensor) -> Tensor:
    return torch.ones_like(tensor)

def zeros_(tensor: Tensor) -> Tensor:
    return torch.zeros_like(tensor)

class BasenjiModel(nn.Module):
    def __init__(self, num_targets, n_channel=4, max_len=128, 
                 conv1kc=64, conv1ks=15, conv1st=1, conv1pd=7, pool1ks=8, pool1st=1 , pdrop1=0.4, #conv_block_1 parameters
                 conv2kc=64, conv2ks=5, conv2st=1, conv2pd=3, pool2ks=4 , pool2st=1, pdrop2=0.4, #conv_block_2 parameters
                 conv3kc=round(64*1.125), conv3ks=5, conv3st=1, conv3pd=3, pool3ks=4 , pool3st=1, pdrop3=0.4, #conv_block_2 parameters
                 convdc = 6, convdkc=32 , convdks=3, debug=False):                 
        super(BasenjiModel, self).__init__()
        
        self.convdc = convdc
        self.debug = debug
        self.num_targets =  num_targets
        ## CNN + dilated CNN
        
        self.conv_block_1 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(n_channel, conv1kc, kernel_size=conv1ks, stride=conv1st, padding=conv1pd, bias=False),
            nn.BatchNorm1d(conv1kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool1ks),
            nn.Dropout(p=0.2))
                
        self.conv_block_2 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(conv1kc, conv2kc, kernel_size=conv2ks, stride=conv2st, padding=conv2pd, bias=False),
            nn.BatchNorm1d(conv2kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool2ks),
            nn.Dropout(p=0.2))
        
        self.conv_block_3 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(conv2kc, round(conv2kc*1.125), kernel_size=conv3ks, stride=conv3st, padding=conv3pd, bias=False),
            nn.BatchNorm1d(conv3kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool3ks),
            nn.Dropout(p=0.2))
        

        self.dilations = nn.ModuleList()
        for i in range(convdc):
            padding = 2**(i)
            self.dilations.append(nn.Sequential(
                upd_GELU(),
                nn.Conv1d(conv3kc, 32, kernel_size=3, padding=padding, dilation=2**i, bias=False),
                nn.BatchNorm1d(32, momentum=0.9, affine=True), 
                upd_GELU(),
                nn.Conv1d(32, 72, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(72, momentum=0.9, affine=True), 
                nn.Dropout(p=0.25)))
            
        self.conv_block_4 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(72, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64, momentum=0.9, affine=True), 
            nn.Dropout(p=0.1)) 
            
        self.conv_block_5 = nn.Sequential(
            upd_GELU(),
            nn.Linear(64, self.num_targets, bias=True),
            nn.Softplus(beta=1, threshold=1000)) 

    
        self.conv_block_1[1].weight.data = self.truncated_normal(self.conv_block_1[1].weight, 0.0, np.sqrt(2/60)) #4
        self.conv_block_2[1].weight.data = self.truncated_normal(self.conv_block_2[1].weight, 0.0, np.sqrt(2/322)) # conv1kc
        self.conv_block_3[1].weight.data = self.truncated_normal(self.conv_block_3[1].weight, 0.0, np.sqrt(2/322)) # conv1kc
        self.conv_block_4[1].weight.data = self.truncated_normal(self.conv_block_4[1].weight, 0.0, np.sqrt(2/72)) # 72
        self.conv_block_5[1].weight.data = self.truncated_normal(self.conv_block_5[1].weight, 0.0, np.sqrt(2/64)) # 64        
        self.conv_block_1[2].weight.data = ones_(self.conv_block_1[2].weight)
        self.conv_block_2[2].weight.data = ones_(self.conv_block_2[2].weight)
        self.conv_block_3[2].weight.data = ones_(self.conv_block_3[2].weight)
        self.conv_block_4[2].weight.data = ones_(self.conv_block_4[2].weight)

        
        for i in range(convdc):
            self.dilations[i][1].weight.data = self.truncated_normal(self.dilations[i][1].weight, 0.0, np.sqrt(2/218)) # 72
            self.dilations[i][-2].weight.data = self.truncated_normal(self.dilations[i][-2].weight, 0.0, np.sqrt(2/32)) # 32
            self.dilations[i][2].weight.data = zeros_(self.dilations[i][2].weight)
            self.dilations[i][-2].weight.data = ones_(self.dilations[i][-2].weight)

    
    def truncated_normal(self, t, mean, std):
        torch.nn.init.normal_(t, mean, std)
        while True:
            cond = torch.logical_or(t < (mean - 2.28*std), t > (mean + 2.28*std))
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t


    def forward(self, seq):
        if self.debug: 
            print (seq.shape)
        seq = self.conv_block_1(seq)
        if self.debug: 
            print ('conv1', seq.shape)
        seq = self.conv_block_2(seq)
        if self.debug: 
            print ('conv2', seq.shape)
        seq = self.conv_block_3(seq)
        if self.debug: 
            print ('conv3', seq.shape)
        for i in range(self.convdc):
            if i == 0:
                y = self.dilations[i](seq)
            if i >= 1:
                y = y.add(self.dilations[i](seq))
            if self.debug: 
                print ('dil', i, self.dilations[i](seq).shape)
        if self.debug:
            print ('y', y.shape)
        res = self.conv_block_4(y)
        if self.debug: 
            print ('res', res.shape)
        res_lin = res.transpose(1, 2)
        if self.debug: 
            print ('res_lin', res_lin.shape)
        res = self.conv_block_5(res_lin)
        if self.debug: 
            print ('res', res.shape)
        return res
        
    def compile(self, device='cpu'):
        self.to(device)