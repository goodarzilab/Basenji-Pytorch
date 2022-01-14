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
    parser.add_argument('-mode', '--mode', type=str, default = 'regression', help='Training mode - regression or classification')
    parser.add_argument('-opt', '--opt' , default = 'Adam', type=str, help='Optimizer')
    parser.add_argument('-batch_size', '--batch_size', type=int, default = 4, help='Batch size')
    parser.add_argument('-num_epochs', '--num_epochs' , default = 1, type=int, help='Number of epochs')
    parser.add_argument('-seq_len', '--seq_len' , default = 131072, type=int, help='Sequence length')
    parser.add_argument('-lr', '--lr' , default = 0.001, type=float, help='Learning rate')
#     parser.add_argument('-device', '--device' , default = 'cuda:2', type=str, help='CUDA name')
    parser.add_argument('-num_targets', '--num_targets' , default = 1, type=int, help='Number of targets to train on')
    parser.add_argument('-target_window', '--target_window' , default = 128, type=int, help='Size of the window that each dataset uses to slice the sequences')
    parser.add_argument('-debug', '--debug' , default = False, type=bool, help='debug mode')
    parser.add_argument('-cut', '--cut' , default = 0.2, type=float, help='Train/Val cut')
    parser.add_argument('-num_workers', '--num_workers' , default = 0, type=int, help='Number of workers used by the dataloader')

    return parser.parse_args()


def main():
    args = get_args()
    param_vals = { 
    "mode": args.mode,
    "optimizer" : args.opt, 
    "init_lr": args.lr, 
    "optimizer_momentum": 0.9, 
    "weight_decay": 1e-3, 
    "loss": "poisson", 
    "num_targets": args.num_targets,
    "lambda_param": 0.001, 
    "ltype":1,
    "clip": 2.,
    "seq_len": args.seq_len,
    "target_window": args.target_window,
    "batch_size": args.batch_size,
    "cut": args.cut,
    "num_workers": args.num_workers,
    "num_epochs": args.num_epochs
    }
    
    targets_memmap_data_dir_cl = '/data/users/goodarzilab/darya/work/Datasets/hg38_targets_memmaps_CL.ATAC'
    targets_memmap_data_dir_pdx = '/data/users/goodarzilab/darya/work/Datasets/hg38_targets_memmaps_PDX.ATAC'
    memmap_data_contigs_dir = '/data/users/goodarzilab/darya/work/Datasets/hg38_memmaps'

    model = BasenjiModel(num_targets=args.num_targets) 
    model.compile(device='cuda')
    trainer = Trainer(param_vals, model, memmap_data_contigs_dir, targets_memmap_data_dir_cl, targets_memmap_data_dir_pdx)
    trainer.train(debug=args.debug)
    
if __name__ == '__main__':
     main()
