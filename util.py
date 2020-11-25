# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:56:46 2020

@author: MSPL
"""
import torch
import torch.nn as nn
#import librosa as lbr
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import shutil
import seaborn as sns
from scipy import signal
import datetime

def estimate_time(remaining_epoch, avg_sec_per_epoch):
    """
    unit time = second
    return (average time per epoch, estimated remaining time, ETA)
    """
    avg_time_per_epoch = datetime.timedelta(avg_sec_per_epoch)
    delta = avg_time_per_epoch * remaining_epoch
    now = datetime.datetime.now()
    ETA = now + delta
    
    return (str(avg_time_per_epoch), str(delta), str(ETA))
     
def reset_vram():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def cuda_setting(gpu_index):
    # return ('device', 'device_index')
    if not torch.cuda.device_count():
        print('This code will be executed on CPU.')
        return ('cpu', [])
    
    if type(gpu_index) == int:
        if gpu_index == -1:
            print('This code will be executed on CPU.')
            return ('cpu', [])
        
        if 0 <= gpu_index <= torch.cuda.device_count() - 1:
            print('This code will be executed on GPU:{}.'.format(gpu_index))
            return ('cuda', [gpu_index])
        else:
            print('GPU index error')
            print('This code will be executed on CPU.')
            return ('cpu', [])
    
    elif type(gpu_index) == list or type(gpu_index) == tuple:
        if len(gpu_index) > torch.cuda.device_count():
            print('GPU indices exceed the number of GPUs.')
            print('This code will be executed on CPU.')
            return ('cpu', [])     
        
        for idx in gpu_index:
            if not (0 <= idx <= torch.cuda.device_count() - 1) or type(idx) != int:
                print('GPU index error')
                print('This code will be executed on CPU.')
                return ('cpu', [])
                
        print('This code will be executed on GPUs:{}.'.format(gpu_index))
        return ('cuda', gpu_index)  

    else:
        print('GPU index error')
        print('This code will be executed on CPU.')
        return ('cpu', [])

def pre_emphasis(signal_batch, emphasis_coef = 0.95, data_type = np.float32):
    signal_batch = signal.lfilter([1, -emphasis_coef], [1], signal_batch)
    signal_batch = signal_batch.astype(data_type)
    
    return signal_batch

def de_emphasis(signal_batch, emphasis_coef = 0.95, data_type = np.float32):
    signal_batch = signal.lfilter([1], [1, -emphasis_coef], signal_batch)
    signal_batch = signal_batch.astype(data_type)
    
    return signal_batch

def check_folder_existance(path_list):
    if type(path_list) == list:
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)
    elif type(path_list) == str:
        if not os.path.exists(path_list):
            os.makedirs(path_list)    

def plot_loss(hist, plot_dir, x_label = 'Iterations'):
    # x_label = 'iterations' or 'epochs'
    x_axis = np.arange(1, len(hist['D_real_loss']) + 1)
     
    color_list = sns.color_palette('deep')
    loss_list = [['D_real_loss', 'D_fake_loss', 'G_GAN_loss'], ['G_L1_loss']]
    
    fig = plt.figure(figsize=(14,12))
    gs = gridspec.GridSpec(nrows = 2, ncols = 1, height_ratios = [2, 1])  
    plt.rcParams['axes.grid'] = True
    
    fig_list = [plt.subplot(gs[0]), plt.subplot(gs[1])]
    
    for subplot_idx in range(len(loss_list)):
        for graph_idx, loss in enumerate(loss_list[subplot_idx]):
            fig_list[subplot_idx].plot(x_axis, hist[loss], linestyle = '-', color = color_list[graph_idx], label = loss)
            plt.xlabel(x_label)
            plt.ylabel('Loss')
            
        fig_list[subplot_idx].legend(loc = 'best')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Loss_{}.png'.format(x_label)))
    plt.close('all')


def extract_chunk(wav, overlap = 0.5):    # extract_chunk; input: a waveform(type: np.array), output: chunks of a waveform
    length_wav = len(wav)
    chunk_size = 16384   #16384 samples at 16kHz sampling rate = 1.024s (chunk size)
    
    hop = int(chunk_size * overlap)
    
    end_idx = np.arange(chunk_size, length_wav + hop, hop)
    if end_idx[-1] > length_wav:
        wav = np.concatenate((wav, np.zeros(end_idx[-1] - length_wav)), axis = 0)
        
    chunk_list = np.zeros((len(end_idx), chunk_size), dtype = np.float32)
    
    for i, idx in enumerate(end_idx):
        chunk_list[i] = wav[i*hop: idx]
    
    return chunk_list # output = [chunk1, chunk2, ...]    

def decay_var(var, var_name, present_epoch, gamma = 1e-5, decay_epoch_interval = 100):
    if (present_epoch + 1) % decay_epoch_interval == 0:
        print('{0} decays. {0} = {1}->{2}'.format(var_name, var, var * gamma))
        return var * gamma
    else:
        return var
    
def weights_init(net):  
    for m in net.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
            if m.bias is not None:
                m.bias.data.fill_(0)
                
def weights_init_xavier(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
                
def Gaussian_Noise(tensor, mean, std):
    return tensor + torch.randn(tensor.size()).to(tensor.device) * std + mean

def cosine_schedule(MAX, MIN, restart_interval, index, shift = False, precision = 8, restart = True):
    # shift: cos(0, pi) -> cos(-pi, 0)
    if shift == False:
        var_list = 0.5 * (MAX - MIN) * np.cos(np.linspace(0, np.pi, restart_interval)) + 0.5 * (MAX + MIN)
    else:
        var_list = 0.5 * (MAX - MIN) * np.cos(np.linspace(-np.pi, 0, restart_interval)) + 0.5 * (MAX + MIN)
    var_list = np.round_(var_list, precision)
    if restart == True or index < restart_interval:
        return var_list[index % restart_interval]
    else:
        return var_list[restart_interval - 1]
def custom_lr_update(optimizer, new_value):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_value