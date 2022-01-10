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
import argparse
import matplotlib.pyplot as plt

import pyBigWig
from scipy.sparse import csc_matrix
import math 

from modules import * 
from model import *


def get_args():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-batch_size', '--batch_size', type=int, default = 8, help='Batch size')
    parser.add_argument('-num_epochs', '--num_epochs' , default = 1, type=int, help='Number of epochs')
    parser.add_argument('-seq_len', '--seq_len' , default = 131072, type=int, help='Sequence length')
#     parser.add_argument('-chroms', '--chroms' , default = 5, type=str, help='Chromosomes to train on')
    parser.add_argument('-lr', '--lr' , default = 0.001, type=float, help='Learning rate')
    parser.add_argument('-opt', '--opt' , default = 'SGD', type=str, help='Optimizer')
    parser.add_argument('-device', '--device' , default = 'cuda:2', type=str, help='CUDA name')
    parser.add_argument('-debug', '--debug' , default = False, type=bool, help='debug mode')
    parser.add_argument('-cut', '--cut' , default = 0.2, type=float, help='Train/Val cut')
    parser.add_argument('-num_gr_train', '--num_gr_train' , default = 900, type=int, help='debug mode')
    parser.add_argument('-num_gr_val', '--num_gr_val' , default = 200, type=int, help='debug mode')

    return parser.parse_args()


def main():
    args = get_args()
    param_vals = { 
    "optimizer" : "Adam", 
    "init_lr": 0.001, 
    "optimizer_momentum": 0.9, 
    "weight_decay": 1e-3, 
    "loss": "poisson", 
    "num_targets": 4,
    "lambda_param": 0.001, 
    "ltype":1,
    "clip": 2.,
    "seq_len": 128*128*8,
    "target_window": 128 * 16,
    "batch_size": 4,
    "cut": 0.8,
    "num_workers": 0,
    "num_epochs": 1
    }
    
    targets_memmap_data_dir_cl = '/data/users/goodarzilab/darya/work/basenji_pytorch/hg38_targets_memmaps_CL.ATAC'
    targets_memmap_data_dir_pdx = '/data/users/goodarzilab/darya/work/basenji_pytorch/hg38_targets_memmaps_PDX.ATAC'
    memmap_data_contigs_dir = '/data/users/goodarzilab/darya/work/basenji_pytorch/hg38_memmaps'

    model = BasenjiModel(num_targets=param_vals.get('num_targets', 3)) 
    model.compile(device='cuda')
    trainer = Trainer(param_vals, model, memmap_data_contigs_dir, targets_memmap_data_dir_cl, targets_memmap_data_dir_pdx)
    trainer.train(debug=True)
    
if __name__ == '__main__':
     main()
