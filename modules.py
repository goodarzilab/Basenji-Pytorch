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

class DNA_Iter(Dataset):
    '''
    A Dataset class. 
    For each chromosome, opens the input and target file, 
    with each __getitem__ call, returns one-hot encoded input and target averaged out over a given window - default is 128. 
    Attributes:         
        target_window (int) - size of the slice from the input and target arrays. Default is 128. 
        seq (mmap) - input file
        tgt_mmap_cl (mmap) - target file (Cordero lab example - CL folder)
        tgt_mmap_pdx (mmap) - target file (Cordero lab example - PDX folder)
        chrom_len (int) - length of the selected chromosome
        nucs (arr) - nucleotides encoded to ints, N -> 5
        switch (bool) - if True, the nucleotide sequence gets reversed 
        switch_func (func) - vectorized function to reverse the nuc sequence 
        num_targets_cl (int) - number of targets 
        num_targets_pdx (int) - number of targets 
        

'''
    def __init__(self, input_name, targets_name_cl, targets_name_pdx, target_window, switch=False):
#         self.input_name = input_name
#         self.targets_name_cl = targets_name_cl
#         self.targets_name_pdx = targets_name_pdx
        self.target_window = target_window
        self.seq = self.read_memmap_input(input_name)
        self.tgt_mmap_cl = self.read_memmap_input(targets_name_cl)
        self.tgt_mmap_pdx = self.read_memmap_input(targets_name_pdx)
        self.chrom_len = self.seq.shape[0]
        self.nucs = np.arange(6.)
        self.switch = switch 
        self.switch_func = np.vectorize(lambda x: x + 1 if (x % 2 == 0) else x - 1)
        self.num_targets_cl = 2 #23
        self.num_targets_pdx = 2 #14

    def __len__(self):
        # length of the dataset is defined as the ration between the full length of the input and the target window
        return int(self.seq.shape[0] / self.target_window)

    def __getitem__(self, idx): 
        # slice the input from the memory map
        seq_subset = self.seq[idx*self.target_window:(idx+1)*self.target_window]
        # if switch=True, reverse the nuc sequence
        if self.switch: 
            seq_subset = self.switch_func(list(reversed(seq_subset)))
        # one-hot encode the input, here compressed row format -> sparse matrix conversion is used
        dta = self.get_csc_matrix(seq_subset)
        max_target_len = max(self.num_targets_cl, self.num_targets_pdx)
        # since the targets are stacked sequentially, we first define the starting point for each
        tgt_lst = np.arange(0, self.chrom_len*max_target_len, self.chrom_len)
        # we then define a single array with indeces of interest that is the same for each target
        arr = np.arange(idx*self.target_window, (idx+1) * self.target_window)
        # we then split the array in chunks of 128, add the "starting points" for each target
        # the array is 3D, which corresponds to each 128-bp chunk in each target 
        ids_cl = np.array([np.split(arr, 128) + tgt for tgt in tgt_lst])
        ids_pdx = np.array([np.split(arr, 128) + tgt for tgt in tgt_lst[:max_target_len]]) #14]])
        # we then calculate the mean accross the 128-bp chunks 
        stacked_means_cl = torch.mean(torch.tensor(np.nan_to_num(np.take(self.tgt_mmap_cl, ids_cl))), dim=1)
        stacked_means_pdx = torch.mean(torch.tensor(np.nan_to_num(np.take(self.tgt_mmap_pdx, ids_pdx))), dim=1)
        # finally, we concatenate the values to get the average for each target
        tgt_window = torch.cat((stacked_means_cl, stacked_means_pdx), dim=0)
        return torch.tensor(dta), tgt_window 

    def read_memmap_input(self, mmap_name):
        '''
        Loads a memory map input
        '''
        seq = np.memmap(mmap_name, dtype='float32',  mode = 'r+') #, shape=(2, self.chrom_seq[self.chrom]))
        return seq

    def get_csc_matrix(self, seq_subset):
        '''
        Converts a compressed row format data to a sparse matrix
        ''' 
        N, M = len(seq_subset), len(self.nucs)
        rows, cols = np.arange(N), seq_subset
        data = np.ones(N, dtype=np.uint8)
        ynew = csc_matrix((data, (rows, cols)), shape=(N, M), dtype=np.uint8)
        return ynew.toarray()[:, :4]



class Trainer(nn.Module):
    '''
    The Trainer class. Handles data loading, model training and evaluation, and visualization. 
    Attributes: 
        param_vals (dict) - a dictionary with the parameters
        model (pytoch model) - a predefined model 
        batch_size (int) - the batch size, pulled from the parameter dictionary
        num_targets (int) - the number of targets that the model is trained on, pulled from the parameter dictionary
        train_losses, valid_losses, train_Rs, valid_Rs, train_R2, valid_R2 (arrs) - arrays that keep track of the loss, Pearson R and R2 
        train_losses_ind (arr) - array that keeps track of individual losses for each target 
    '''
    def __init__(self, param_vals, model, memmap_data_contigs_dir, memmap_data_targets_dir_cl, memmap_data_targets_dir_pdx):
        super(Trainer, self).__init__()
    
        self.param_vals = param_vals
        self.model = model 
        
        self.train_losses, self.valid_losses, self.train_Rs, self.valid_Rs, self.train_R2, self.valid_R2 = [], [], [], [], [], []
        self.train_losses_ind = [[] for i in range(self.param_vals.get('num_targets', 1))]
        self.optim_step = 0
        
        self.batch_size = self.param_vals.get('batch_size', 8)
        self.num_targets = self.param_vals.get('num_targets', 1)
        self.make_optimizer()
        self.init_loss()
        self.make_dsets(memmap_data_contigs_dir, memmap_data_targets_dir_cl, memmap_data_targets_dir_pdx)
        print ('init dsets')

    def make_optimizer(self): 
        '''
        Initializes the optimizer
        '''
        if self.param_vals["optimizer"]=="Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.param_vals["init_lr"])
        if self.param_vals["optimizer"]=="AdamW":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.param_vals["init_lr"])
        if self.param_vals["optimizer"]=="SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.param_vals["init_lr"]) #, momentum = self.param_vals["optimizer_momentum"])
        if self.param_vals["optimizer"]=="Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.param_vals["init_lr"], weight_decay = self.param_vals["weight_decay"])
    
    def make_dsets(self, input_files_dir, target_files_dir_cl, target_files_dir_pdx):
        '''
        Initizes the datasets
        '''
        cut = self.param_vals.get('cut', .8)
        np.random.seed(42)
        # select the files with the .dta extension
        chroms_list = [file.split('_')[0] for file in os.listdir(input_files_dir) if file.split('.')[-1] == 'dta']
        # shuffle the files
        np.random.shuffle(chroms_list)
        # create the input and target file lists for training and validation datasets 
        input_list = np.hstack([[file for file in os.listdir(input_files_dir) if file.split('_')[0] == chrom] for chrom in chroms_list])
        targets_cl_list = np.hstack([[file for file in os.listdir(target_files_dir_cl) if file.split('_')[0] == chrom] for chrom in chroms_list])
        targets_pdx_list = np.hstack([[file for file in os.listdir(target_files_dir_pdx) if file.split('_')[0] == chrom] for chrom in chroms_list])
        
        val_input_files = input_list[int(len(input_list)*cut):]
        val_target_files_cl = targets_cl_list[int(len(targets_cl_list)*cut):]
        val_target_files_pdx = targets_pdx_list[int(len(targets_pdx_list)*cut):]

        
        train_input_files = input_list[:int(len(input_list)*cut)]
        train_target_cl_files = targets_cl_list[:int(len(targets_cl_list)*cut)]
        train_target_pdx_files = targets_pdx_list[:int(len(targets_pdx_list)*cut)]

        # concatenate the datasets defined for each chromosome 
        self.valid_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, val_input_files[i]), 
                                                  os.path.join(target_files_dir_cl, val_target_files_cl[i]), 
                                                  os.path.join(target_files_dir_pdx, val_target_files_pdx[i]), 
                                                  self.param_vals.get('target_window', 128)
                                                  ) for i in range(len(val_input_files))])
        
        self.training_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, train_input_files[i]), 
                                                     os.path.join(target_files_dir_cl, train_target_cl_files[i]), 
                                                     os.path.join(target_files_dir_pdx, train_target_pdx_files[i]), 
                                                     self.param_vals.get('target_window', 128),
                                                     switch=False) for i in range(len(train_input_files))])
        
        self.training_dset_augm = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, train_input_files[i]), 
                                                          os.path.join(target_files_dir_cl, train_target_cl_files[i]),  
                                                          os.path.join(target_files_dir_pdx, train_target_pdx_files[i]), 
                                                          self.param_vals.get('target_window', 128),
                                                          switch=True) for i in range(len(train_input_files))])

        
    def make_loaders(self, augm):
        '''
        Initializes three dataloaders: training, validation, and training with reversed nucleotides. 
        '''
        # the batch size for the dataloaders is defined as (seq_len * batch_size) / target_window
        batch_size = int((self.param_vals.get('seq_len', 128*128*8)*self.param_vals.get('batch_size', 8)) / self.param_vals.get('target_window', 128))
        num_workers = self.param_vals.get('num_workers', 8)
        if augm: 
            train_loader = DataLoader(dataset=self.training_dset_augm, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        else: 
            train_loader = DataLoader(dataset=self.training_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(dataset=self.valid_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        return train_loader, val_loader
    
    
    def decayed_learning_rate(self, step, initial_learning_rate, decay_rate=0.96, decay_steps=100000):
        '''
        Define the decayed learning rate.
        '''
        return initial_learning_rate * math.pow(decay_rate, (step / decay_steps))
    
    def upd_optimizer(self, optim_step):
        '''
        Update the optimizer given the decayed learning rate calculated above. 
        '''
        decayed_lr = self.decayed_learning_rate(optim_step, initial_learning_rate=self.param_vals["init_lr"])
        for g in self.optimizer.param_groups:
            g['lr'] = decayed_lr 

        
    def init_loss(self, reduction="sum"):
        '''
        Initializes the losses. 
        '''
        if self.param_vals["loss"]=="mse":
            self.loss_fn = F.mse_loss
        if self.param_vals["loss"]=="poisson":
            self.loss_fn = torch.nn.PoissonNLLLoss(log_input=False, reduction=reduction)
    
    def get_input(self, batch):
        '''
        Returns X and y for each batch returned by a dataloader. 
        '''
        batch_size = self.param_vals.get('batch_size', 8)
        num_targets = self.param_vals.get('num_targets', 1)
        seq_X,y = batch
        # reshape the input into [batch_size, 4, seq_len] format
        X_reshape = torch.stack(torch.chunk(torch.transpose(seq_X.reshape(seq_X.shape[0]*seq_X.shape[1], 4), 1, 0), batch_size, dim=1)).type(torch.FloatTensor).cuda()
        # this can be replaces by drop_last=True parameter in the dataloaders to drop the batches in the last iteration for each epoch
        # ensures that the batch is full size 
        if X_reshape.shape[-1] == self.param_vals.get('seq_len', 128*128*8):
            # reshape the target into [batch_size, 1024, num_targets]
            y =  torch.stack(torch.chunk(y, batch_size, dim=0)).view(batch_size, 1024, num_targets).type(torch.FloatTensor).cuda()
            y = F.normalize(y, dim=1)            
            return X_reshape, y
        else:
            return np.array([0]), np.array([0])

    def plot_results(self, y, out, num_targets):
        '''
        Plots the predictions vs the true values. 
        '''
        for i in range(num_targets):
            ys = y[:, :, i].flatten().cpu().numpy()
            preds = out[:, :, i].flatten().detach().cpu().numpy()
            plt.plot(np.arange(len(ys.flatten())), ys.flatten(), label='True')
            plt.plot(np.arange(len(preds.flatten())), preds.flatten(), label='Predicted', alpha=0.5)
            plt.legend()
            plt.show()        
    
    def train(self, debug):
        '''
        Main training loop
        '''
        print('began training')
        for epoch in range(self.param_vals.get('num_epochs', 10)):
            if epoch % 2 == 0: 
                augm = False
            else: 
                augm = True
            train_loader, val_loader = self.make_loaders(augm)
            print(len(train_loader), len(val_loader))
            for batch_idx, batch in enumerate(train_loader):
                print_res, plot_res = False, False
                self.model.train()
                x, y = self.get_input(batch)
                if (debug): 
                    print (x.shape, y.shape)
                if x.shape[0] != 1: 
                    self.optimizer.zero_grad()
                    if batch_idx%10==0:
                        print_res = True
                        if batch_idx%300==0:
                            plot_res = True
                    self.train_step(x, y, print_res, plot_res, epoch, batch_idx, train_loader)
                    print_res, plot_res = False, False
#             print(self.train_R2)
                             
            if val_loader:
                print_res, plot_res = False, False
                self.model.eval()
                for batch_idx, batch in enumerate(val_loader):
                    print_res, plot_res = False, False 
                    x, y = self.get_input(batch)
                    if x.shape[0] != 1: 
                        if batch_idx%10==0:
                            print_res = True
                            if batch_idx%300==0:
                                plot_res = True
                        self.eval_step(x, y, print_res, plot_res, epoch, batch_idx, val_loader) 
                        print_res, plot_res = False, False 

            train_arrs = np.array([self.train_losses, self.train_Rs, self.train_R2])
            val_arrs = np.array([self.valid_losses, self.valid_Rs, self.valid_R2])
            self.plot_metrics(epoch+1, train_arrs, val_arrs)
            self.plot_ind_loss(epoch+1, self.train_losses_ind)



    def train_step(self, x, y, print_res, plot_res, epoch, batch_idx, train_loader):
        '''
        Define each training step
        '''

        with torch.cuda.amp.autocast():
            out = self.model(x).view(y.shape)
            loss = 0
            # calculate loss for each target
            for i in range(y.shape[-1]):
                loss_ = self.loss_fn(out[:, :, i],y[:, :, i])
                self.train_losses_ind[i].append(loss_.data.item())
                loss += loss_
            # if the regularization is required, update the loss
            if self.param_vals.get('lambda_param', None): 
                loss = self.regularize_loss(self.param_vals["lambda_param"], self.param_vals["ltype"], self.model, loss)
        # calculate the Pearson R and R2 
        R, r2 = self.calc_R_R2(y, out, self.num_targets)
        
        # backpropagate the loss
        loss.backward()
        # clip the gradient if required
        if self.param_vals.get('clip', None): 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.param_vals["clip"])
        
        # update the optimizer
        self.optimizer.step()
        self.optim_step += 1
        self.upd_optimizer(self.optim_step)
        
        # record the values for loss, Pearson R, and R2
        self.train_losses.append(loss.data.item())
        self.train_Rs.append(R.item())
        self.train_R2.append(r2.item())
        if print_res: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}\tR2: {:.6f}'.format(
                          epoch, batch_idx, len(train_loader), int(100. * batch_idx / len(train_loader)),
                          loss.item(), R.item(), r2.item()))
        if plot_res: 
            print (torch.sum(y).item(), torch.sum(out).item())
            self.plot_results(y, out, self.num_targets)

    def eval_step(self, x, y, print_res, plot_res, epoch, batch_idx, val_loader):
        '''
        Define each evaluation step
        '''

        out = self.model(x).view(y.shape)
        loss = 0
        for i in range(y.shape[-1]):
            loss_ = self.loss_fn(out[:, :, i],y[:, :, i])
            loss += loss_

#         loss = self.loss_fn(out,y)
        R, r2 = self.calc_R_R2(y, out, self.num_targets)
        self.valid_losses.append(loss.data.item())
        self.valid_Rs.append(R.item())
        self.valid_R2.append(r2.item())                
        
        if print_res: 
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}\tR2: {:.6f}'.format(
                          epoch, batch_idx, len(val_loader), int(100. * batch_idx / len(val_loader)),
                          loss.item(), R.item(), r2.item()))
        if plot_res: 
            self.plot_results(y, out, self.num_targets)

    def mean_arr(self, num_epochs, arr):
        num_iter = int(len(arr) / num_epochs)
        mean_train_arr = [np.mean(arr[i*num_iter:(i+1)*num_iter]) for i in range(num_epochs)]
        return mean_train_arr
            
    def plot_metrics(self, num_epochs, train_arrs, val_arrs): 
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        for i in range(3):
            mean_train_arr = self.mean_arr(num_epochs, train_arrs[i])
            mean_val_arr = self.mean_arr(num_epochs, val_arrs[i])
            axs[i].plot(np.arange(num_epochs-1), mean_train_arr[1:], label='Train')
            axs[i].plot(np.arange(num_epochs-1), mean_val_arr[1:], label='Val')
        fig.tight_layout()
        plt.show()    
        
    def plot_ind_loss(self, num_epochs, train_arrs_ind):
        '''
        Plots individual losses for 4 targets side by side
        '''
#         num_targets = self.param_vals.get('num_targets', 1)
        num_targets = 4

        fig, axs = plt.subplots(nrows=1, ncols=num_targets+1, figsize=(15, 3))
        for i in range(num_targets):
            mean_train_arr = self.mean_arr(num_epochs, train_arrs_ind[i])
            axs[num_targets].plot(np.arange(num_epochs-1), mean_train_arr[1:], label='Train')
            axs[i].plot(np.arange(num_epochs-1), mean_train_arr[1:], label='Train')
        fig.tight_layout()
        plt.show()    


        
        
    def calc_R_R2(self, y_true, y_pred, num_targets, device='cuda:0'):
        '''
        Handles the Pearson R and R2 calculation
        '''
        product = torch.sum(torch.multiply(y_true, y_pred), dim=1)
        true_sum = torch.sum(y_true, dim=1)
        true_sumsq = torch.sum(torch.square(y_true), dim=1)
        pred_sum = torch.sum(y_pred, dim=1)
        pred_sumsq = torch.sum(torch.square(y_pred), dim=1)
        count = torch.sum(torch.ones(y_true.shape), dim=1).to(device)
        true_mean = torch.divide(true_sum, count)
        true_mean2 = torch.square(true_mean)

        pred_mean = torch.divide(pred_sum, count)
        pred_mean2 = torch.square(pred_mean)

        term1 = product
        term2 = -torch.multiply(true_mean, pred_sum)
        term3 = -torch.multiply(pred_mean, true_sum)
        term4 = torch.multiply(count, torch.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = true_sumsq - torch.multiply(count, true_mean2)
        pred_var = pred_sumsq - torch.multiply(count, pred_mean2)
        pred_var = torch.where(torch.greater(pred_var, 1e-12), pred_var, np.inf*torch.ones(pred_var.shape).to(device))

        tp_var = torch.multiply(torch.sqrt(true_var), torch.sqrt(pred_var))

        correlation = torch.divide(covariance, tp_var)
        correlation = correlation[~torch.isnan(correlation)]
        correlation_mean = torch.mean(correlation)
        total = torch.subtract(true_sumsq, torch.multiply(count, true_mean2))
        resid1 = pred_sumsq
        resid2 = -2*product 
        resid3 = true_sumsq
        resid = resid1 + resid2 + resid3 
        r2 = torch.ones_like(torch.tensor(num_targets)) - torch.divide(resid, total)
        r2 = r2[~torch.isinf(r2)]
        r2_mean = torch.mean(r2)
        return correlation_mean, r2_mean


        
    def regularize_loss(self, lambda1, ltype, net, loss):
        '''
        Handles regularization for each conv block.
            ltype values: 
                1, 2 - L1 and L2 regularizations 
                3 - gradient clipping 
               
        '''
        if ltype == 3:
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_1.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_2.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_3.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_4.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                        net.conv_block_5.parameters(), lambda1)
                for i in range(len(net.dilations)):
                    torch.nn.utils.clip_grad_norm_(
                        net.dilations[i].parameters(), lambda1)

        else:      
            l0_params = torch.cat(
                [x.view(-1) for x in net.conv_block_1[1].parameters()])
            l1_params = torch.cat(
                [x.view(-1) for x in net.conv_block_2[1].parameters()])
            l2_params = torch.cat(
                [x.view(-1) for x in net.conv_block_3[1].parameters()])
            l3_params = torch.cat(
                [x.view(-1) for x in net.conv_block_4[1].parameters()])
            l4_params = torch.cat(
                    [x.view(-1) for x in net.conv_block_5[1].parameters()])
            dil_params = []
            for i in range(len(net.dilations)):
                dil_params.append(torch.cat(
                    [x.view(-1) for x in net.dilations[i][1].parameters()]))

        if ltype in [1, 2]:
            l1_l0 = lambda1 * torch.norm(l0_params, ltype)
            l1_l1 = lambda1 * torch.norm(l1_params, ltype)
            l1_l2 = lambda1 * torch.norm(l2_params, ltype)
            l1_l3 = lambda1 * torch.norm(l3_params, ltype)
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            l1_l4 = lambda1 * torch.norm(l4_params, 2)
            dil_norm = []
            for d in dil_params:
                dil_norm.append(lambda1 * torch.norm(d, ltype))  
            loss = loss + l1_l0 + l1_l1 + l1_l2 + l1_l3 + l1_l4 + torch.stack(dil_norm).sum()

        elif ltype == 4:
            l1_l0 = lambda1 * torch.norm(l0_params, 1)
            l1_l1 = lambda1 * torch.norm(l1_params, 1)
            l1_l2 = lambda1 * torch.norm(l2_params, 1)
            l1_l3 = lambda1 * torch.norm(l3_params, 1)
            l2_l0 = lambda1 * torch.norm(l0_params, 2)
            l2_l1 = lambda1 * torch.norm(l1_params, 2)
            l2_l2 = lambda1 * torch.norm(l2_params, 2)
            l2_l3 = lambda1 * torch.norm(l3_params, 2)
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            l2_l4 = lambda1 * torch.norm(l4_params, 2)
            dil_norm1, dil_norm2 = [], []
            for d in dil_params:
                dil_norm1.append(lambda1 * torch.norm(d, 1))  
                dil_norm2.append(lambda1 * torch.norm(d, 2))  

            loss = loss + l1_l0 + l1_l1 + l1_l2 +\
                    l1_l3 + l1_l4 + l2_l0 + l2_l1 +\
                    l2_l2 + l2_l3 + l2_l4 + \
                torch.stack(dil_norm1).sum() + torch.stack(dil_norm2).sum()
        return loss

    def save_model(self, model, filename):
        '''
        Handles model saving
        '''
        torch.save(model.state_dict(), filename)

    def load_model(self, model, filename):
        '''
        Handles model loading
        '''
        model.load_state_dict(torch.load(filename))    