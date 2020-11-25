# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:17:15 2020

@author: MSPL
"""

#from model import Generator, Discriminator
import copy
from util import *
import time
import datetime
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa as lbr
from torch.utils.data import DataLoader
from dataloader import Valentini_VCTK
import torch.autograd as autograd
# from torchsummary import summary
import random

class Core(object):
    def __init__(self, Generator, Discriminator, **kwargs):
        # Hyper-parameters
        self.hyper_param = {'epochs': kwargs['epochs'],
                            'batch_size': kwargs['epochs'],
                            'reduced_model': kwargs['reduced_model'],
                            'G_lr': kwargs['G_lr'],
                            'D_lr': kwargs['D_lr'],
                            'clean_only': kwargs['clean_only'],
                            'G_lr_cos_scheduler': kwargs['G_lr_cos_scheduler'],
                            'D_lr_cos_scheduler': kwargs['D_lr_cos_scheduler'],
                            'G_lr_cos_scheduler_opt': kwargs['G_lr_cos_scheduler_opt'],
                            'D_lr_cos_scheduler_opt': kwargs['D_lr_cos_scheduler_opt'],
                            'gradient_penalty': kwargs['gradient_penalty'],
                            'spectral_normalization': kwargs['spectral_normalization'],
                            'D_batch_normalization': kwargs['D_batch_normalization'],
                            'lambda_G': kwargs['lambda_G'],
                            'lambda_gp': kwargs['lambda_gp'],
                            'seed': kwargs['seed']
                            }
        # Options
        self.opt = {'info_num_per_epoch': kwargs['info_num_per_epoch'],
                    'loaded_epoch': kwargs['loaded_epoch'],
                    'save_epoch_interval': kwargs['save_epoch_interval'],
                    'model_name': kwargs['model_name'],
                    'sampling_rate': kwargs['sampling_rate'],
                    'train': kwargs['train']
                    }

        # directory
        self.dir = {'model': os.path.join(os.path.abspath(''), 'models'),
                    'sample': os.path.join(os.path.abspath(''), 'samples'),
                    'inference': os.path.join(os.path.abspath(''), 'inference'),
                    'plot': os.path.join(os.path.abspath(''), 'plot'),
                    'inf_sample': os.path.join(os.path.abspath(''), 'inference', 'sample')
                    }
        # CUDA
        self.device, self.device_idx = cuda_setting(kwargs['device_id'])
        if len(self.device_idx) == 1:
            self.device = self.device + ':{}'.format(self.device_idx[0])
        
        # Seed to reproducibility
        random.seed(self.hyper_param['seed'])
        np.random.seed(self.hyper_param['seed'])
        torch.manual_seed(self.hyper_param['seed'])
        if self.device != 'cpu':
            torch.cuda.manual_seed_all(self.hyper_param['seed'])

        # Generator
        self.G = Generator(reduced = self.hyper_param['reduced_model']).to(self.device)
         
        if self.opt['train'] == True: # If Generator has trained(self.opt['train'] == False), Discriminator and optimizer are disabled.
            # Load Hyper-parameters
            self.load_hyper_param(epoch = self.opt['loaded_epoch'])
            self.hyper_param['reduced_model'] = kwargs['reduced_model'] # for the backward_compatibility
            
            # Discriminator
            self.D = Discriminator(reduced = self.hyper_param['reduced_model'],
                                   batch_norm = self.hyper_param['D_batch_normalization'],
                                   spectral_norm = self.hyper_param['spectral_normalization']
                                   ).to(self.device)     
            # Optimizer
#            self.G_opt = optim.AdamW(self.G.parameters(), lr = self.hyper_param['G_lr'], weight_decay = 1e-3)
#            self.D_opt = optim.AdamW(self.D.parameters(), lr = self.hyper_param['D_lr'], weight_decay = 1e-3)
            self.G_opt = optim.RMSprop(self.G.parameters(), lr = self.hyper_param['G_lr'])
            self.D_opt = optim.RMSprop(self.D.parameters(), lr = self.hyper_param['D_lr'])
            # Learning rate scheduler
            self.G_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer = self.G_opt,
                    step_size = kwargs['G_lr_scheduler_opt']['step_size'],
                    gamma = kwargs['G_lr_scheduler_opt']['gamma'],
                    last_epoch = kwargs['G_lr_scheduler_opt']['last_epoch']) if kwargs['G_lr_scheduler'] == True else None
            self.D_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer = self.D_opt,
                    step_size = kwargs['D_lr_scheduler_opt']['step_size'],
                    gamma = kwargs['D_lr_scheduler_opt']['gamma'],
                    last_epoch = kwargs['D_lr_scheduler_opt']['last_epoch']) if kwargs['D_lr_scheduler'] == True else None
        
            # Loss functions
            self.L1_loss = nn.L1Loss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)

            # History dictionary                                                 
            self.train_hist_iter = {'D_real_loss': [],
                                    'D_fake_loss': [],
                                    'G_GAN_loss': [],
                                    'G_L1_loss': []
                                    }
            self.train_hist_epoch = {'D_real_loss': [],
                                     'D_fake_loss': [],
                                     'G_GAN_loss': [],
                                     'G_L1_loss': [],
                                     'per_epoch_time': []
                                     }
        
            check_folder_existance(list(self.dir.values()))

            # Load models and hist
            self.load_discriminator(epoch = self.opt['loaded_epoch'], map_loc = self.device)
            self.load_hist(epoch = self.opt['loaded_epoch'])  
            
        self.load_generator(epoch = self.opt['loaded_epoch'], map_loc = self.device)
        
        # Multi GPU environment
        if len(self.device_idx) > 1:
            self.G = nn.DataParallel(self.G, device_ids = self.device_idx)
            if self.opt['train'] == True:
                self.D = nn.DataParallel(self.D, device_ids = self.device_idx)
  
    


        
    def train_codec_enhancement(self):
        self.Dataset = Valentini_VCTK(train = True, pre_emphasis = True, clean_only = self.hyper_param['clean_only'])
        self.DataLoader = DataLoader(self.Dataset,
                                     batch_size = self.hyper_param['batch_size'],
                                     drop_last = True,
                                     shuffle = True,
                                     pin_memory = False,
                                     num_workers = 0)
        
        # information print setting
        self.iteration_num = self.DataLoader.dataset.__len__() // self.hyper_param['batch_size']
        self.info_interval = np.linspace(0, self.iteration_num, num = self.opt['info_num_per_epoch'] + 1, endpoint = False, dtype = np.int16)[1:]
        
        print('###################')
        print('Dataset Information')
        print('Sample Num: {}'.format(self.DataLoader.dataset.__len__()))
        print('Batch Size: {}'.format(self.hyper_param['batch_size']))
        print('1 Epoch = {}'.format(self.iteration_num))
        print('###################')
              
        label_real, label_fake = torch.ones(self.hyper_param['batch_size'], 1).to(self.device), torch.zeros(self.hyper_param['batch_size'], 1).to(self.device)
        epoch_hyper_param = copy.deepcopy(self.hyper_param)
        for epoch in range(self.opt['loaded_epoch'], self.hyper_param['epochs']):
            self.G.train()
            # Varying Hyper-parameters (if Hyper-parameter is varing, use epoch_hyper_param; manually change values)                
            epoch_hyper_param['lambda_G'] = cosine_schedule(
                    MAX = self.hyper_param['lambda_G'],
                    MIN = self.hyper_param['lambda_G'] * 1e-5,
                    restart_interval = 100,
                    index = epoch,
                    shift = False,
                    restart = False
                    )
            if self.hyper_param['G_lr_cos_scheduler'] == True:
                epoch_hyper_param['G_lr'] = cosine_schedule(
                        MAX = 5e-5,
                        MIN = 1e-6,
                        restart_interval = self.hyper_param['G_lr_cos_scheduler_opt']['restart_interval'],
                        index = epoch,
                        shift = self.hyper_param['G_lr_cos_scheduler_opt']['shift'],
                        )
                custom_lr_update(self.G_opt, epoch_hyper_param['G_lr'])
                
            if self.hyper_param['D_lr_cos_scheduler'] == True:
                epoch_hyper_param['D_lr'] = cosine_schedule(
                        MAX = 5e-5,
                        MIN = 1e-6,
                        restart_interval = self.hyper_param['D_lr_cos_scheduler_opt']['restart_interval'],
                        index = epoch,
                        shift = self.hyper_param['D_lr_cos_scheduler_opt']['shift'],
                        )
                custom_lr_update(self.D_opt, epoch_hyper_param['D_lr'])                 
            # For saving epoch_history
            epoch_hist = {'D_real_loss': [],
                          'D_fake_loss': [],
                          'G_GAN_loss': [],
                          'G_L1_loss': []
                          }
            epoch_start_time = time.time()
            print('Epoch [{:4d}/{:4d}]'.format(epoch + 1, self.hyper_param['epochs']))
            print('Generator LR: {}, Discriminator LR: {}, lambda_G: {}'.format(epoch_hyper_param['G_lr'], epoch_hyper_param['D_lr'], epoch_hyper_param['lambda_G']))

            for iteration, (original, decoded) in enumerate(self.DataLoader):
                original, decoded = original.to(self.device), decoded.to(self.device)
                if iteration == self.iteration_num:
                    break
#                if self.noise_std > self.noise_std_bound:
#                    original = Gaussian_Noise(original, mean = 0, std = self.noise_std)
                # Noise
                if self.hyper_param['reduced_model'] == False:
                    z = torch.randn((self.hyper_param['batch_size'], 1024, 8)).to(self.device)
                else:
                    z = torch.randn((self.hyper_param['batch_size'], 1024, 16)).to(self.device)
                # Enhanced
                enhanced = self.G(decoded, z)
                # Real pair and fake pair
                real_pair = self.D(original, decoded)
                fake_pair = self.D(enhanced, decoded)  

                self.D_opt.zero_grad()
                
                if self.hyper_param['gradient_penalty'] == True:
                    # Gradient panelty
                    alpha = torch.rand(self.hyper_param['batch_size'], 1, 1).to(self.device)
                    interpolated = alpha * original + (1 - alpha) * enhanced
                    interpolated = interpolated.to(self.device)
                    interpolated.requires_grad_(True)
                    
                    d_interpolated = self.D(interpolated, decoded)
                    grad_penalty = autograd.grad(outputs = d_interpolated,
                                            inputs = interpolated,
                                            grad_outputs = label_real,
                                            create_graph = True,
                                            retain_graph = True,
                                            only_inputs = True
                                            )[0]
                    grad_penalty = self.hyper_param['lambda_gp'] * ((grad_penalty.norm(2, 1) - 1) ** 2).mean()

                # Discriminator loss
                D_real_loss = self.MSE_loss(real_pair, label_real) # D_real_loss = 0.5 * torch.mean((real_pair - 1.0) ** 2)
                D_fake_loss = self.MSE_loss(fake_pair, label_fake) # D_fake_loss = 0.5 * torch.mean(fake_pair ** 2)
                
                if self.hyper_param['gradient_penalty'] == True:
                    D_loss = 0.5 * (D_real_loss + D_fake_loss) + grad_penalty
                else:
                    D_loss = 0.5 * (D_real_loss + D_fake_loss)
                
                D_loss.backward()
                self.D_opt.step()

                epoch_hist['D_real_loss'].append(D_real_loss.item()) # tensor.item: returns the value of this tensor as a python data.
                epoch_hist['D_fake_loss'].append(D_fake_loss.item())
                
                # Generator loss
                self.G_opt.zero_grad()

                enhanced = self.G(decoded, z)
                fake_pair = self.D(enhanced, decoded)
                
                G_GAN_loss = self.MSE_loss(fake_pair, label_real) # G_GAN_loss = 0.5 * torch.mean((fake_pair - 1.0) ** 2)
                G_L1_loss = self.L1_loss(enhanced, original)
                
                G_loss = 0.5 * G_GAN_loss + epoch_hyper_param['lambda_G'] * G_L1_loss
                G_loss.backward()
                self.G_opt.step()
                
                epoch_hist['G_GAN_loss'].append(G_GAN_loss.item())
                epoch_hist['G_L1_loss'].append(G_L1_loss.item())
                    
                if np.any(self.info_interval == (iteration + 1)) == True:
                    print("[{:4d}/{:4d} ({:3d}%)]: D_loss: {:.4f}, D_r_loss: {:.4f}, D_f_loss: {:.4f}, G_loss: {:.4f}, G_GAN_loss: {:.4f}, G_L1_loss: {:.4f}, D(o, d): {:.3f}, D(G(d, z), d): {:.3f}"
                          .format(iteration + 1,
                                  self.iteration_num,
                                  ((iteration + 1)*100 // self.iteration_num),
                                  D_loss.item(),
                                  D_real_loss.item(),
                                  D_fake_loss.item(),
                                  G_loss.item(),
                                  G_GAN_loss.item(),
                                  G_L1_loss.item(),
                                  real_pair.mean().item(),
                                  fake_pair.mean().item()))
                    
            # record the train histories
            self.train_hist_iter['D_real_loss'].extend(epoch_hist['D_real_loss']) 
            self.train_hist_iter['D_fake_loss'].extend(epoch_hist['D_fake_loss']) 
            self.train_hist_iter['G_GAN_loss'].extend(epoch_hist['G_GAN_loss'])
            self.train_hist_iter['G_L1_loss'].extend(epoch_hist['G_L1_loss'])
            
            self.train_hist_epoch['D_real_loss'].append(np.array(epoch_hist['D_real_loss']).mean()) # np.array for calculating mean value
            self.train_hist_epoch['D_fake_loss'].append(np.array(epoch_hist['D_fake_loss']).mean())
            self.train_hist_epoch['G_GAN_loss'].append(np.array(epoch_hist['G_GAN_loss']).mean())
            self.train_hist_epoch['G_L1_loss'].append(np.array(epoch_hist['G_L1_loss']).mean())
            self.train_hist_epoch['per_epoch_time'].append(time.time() - epoch_start_time)
            
            avg_epoch_time = int(np.mean(self.train_hist_epoch['per_epoch_time']))
            print("  Avg. 1 epoch time: [%s] / Est. remaining time: [%s]" % (str(datetime.timedelta(seconds = avg_epoch_time)),
                                                                                        str(datetime.timedelta(seconds = (self.hyper_param['epochs'] - epoch - 1)*avg_epoch_time))))
            
            if (epoch + 1) % self.opt['save_epoch_interval'] == 0:
                self.save(epoch)
            
            # learning rate scheduler
            if self.G_scheduler != None:
                self.G_scheduler.step()
            if self.D_scheduler != None:
                self.D_scheduler.step()
                
            self.inference(epoch = epoch + 1,
                           input_file = os.path.join(self.dir['inf_sample'], 'decoded.wav'),
                           output_file = os.path.join(self.dir['inference'], 'enhanced_epoch_{}.wav'.format(epoch + 1))
                           )
            
            # plot losses
            plot_loss(hist = self.train_hist_epoch, plot_dir = self.dir['plot'], x_label = 'Epochs')
            plot_loss(hist = self.train_hist_iter, plot_dir = self.dir['plot'], x_label = 'Iterations')

                
        print('Training Finish.')

    def train_codec_approximate(self):
        self.Dataset = Valentini_VCTK(train = True, pre_emphasis = True)
        self.DataLoader = DataLoader(self.Dataset,
                                     batch_size = self.hyper_param['batch_size'],
                                     drop_last = True,
                                     shuffle = True,
                                     pin_memory = False,
                                     num_workers = 0)
        
        # information print setting
        self.iteration_num = self.DataLoader.dataset.__len__() // self.hyper_param['batch_size']
        self.info_interval = np.linspace(0, self.iteration_num, num = self.opt['info_num_per_epoch'] + 1, endpoint = False, dtype = np.int16)[1:]
        
        print('###################')
        print('Dataset Information')
        print('Sample Num: {}'.format(self.DataLoader.dataset.__len__()))
        print('Batch Size: {}'.format(self.hyper_param['batch_size']))
        print('1 Epoch = {}'.format(self.iteration_num))
        print('###################')
              
        label_real, label_fake = torch.ones(self.hyper_param['batch_size'], 1).to(self.device), torch.zeros(self.hyper_param['batch_size'], 1).to(self.device)
        
        epoch_hyper_param = copy.deepcopy(self.hyper_param)
        for epoch in range(self.opt['loaded_epoch'], self.hyper_param['epochs']):
            self.G.train()
            # Varying Hyper-parameters (if Hyper-parameter is varing, use epoch_hyper_param; manually change values)
            if self.hyper_param['G_lr_cos_scheduler'] == True:
                epoch_hyper_param['G_lr'] = cosine_schedule(
                        MAX = 1e1 * self.hyper_param['G_lr'],
                        MIN = 1e-1 * self.hyper_param['G_lr'],
                        restart_interval = self.hyper_param['G_lr_cos_scheduler_opt']['restart_interval'],
                        index = epoch,
                        shift = self.hyper_param['G_lr_cos_scheduler_opt']['shift'],
                        )
                custom_lr_update(self.G_opt, epoch_hyper_param['G_lr'])
                
            if self.hyper_param['D_lr_cos_scheduler'] == True:
                epoch_hyper_param['D_lr'] = cosine_schedule(
                        MAX = 1e1 * self.hyper_param['D_lr'],
                        MIN = 1e-1 * self.hyper_param['D_lr'],
                        restart_interval = self.hyper_param['D_lr_cos_scheduler_opt']['restart_interval'],
                        index = epoch,
                        shift = self.hyper_param['D_lr_cos_scheduler_opt']['shift'],
                        )
                custom_lr_update(self.D_opt, epoch_hyper_param['D_lr'])               
            epoch_hyper_param['lambda_G'] = cosine_schedule(
                    MAX = self.hyper_param['lambda_G'],
                    MIN = self.hyper_param['lambda_G'] * 1e-5,
                    restart_interval = self.hyper_param['epochs'],
                    index = epoch,
                    shift = False
                    )
            
            # For saving epoch_history
            epoch_hist = {'D_real_loss': [],
                          'D_fake_loss': [],
                          'G_GAN_loss': [],
                          'G_L1_loss': []
                          }
            epoch_start_time = time.time()
            print('Epoch [{:4d}/{:4d}]'.format(epoch + 1, self.hyper_param['epochs']))
            print('Warmstart Generator LR: {}, Warmstart Discriminator LR: {}, lambda_G: {}'.format(epoch_hyper_param['G_lr'], epoch_hyper_param['D_lr'], epoch_hyper_param['lambda_G']))

            for iteration, (original, decoded) in enumerate(self.DataLoader):
                original, decoded = original.to(self.device), decoded.to(self.device)
                if iteration == self.iteration_num:
                    break
#                if self.noise_std > self.noise_std_bound:
#                    original = Gaussian_Noise(original, mean = 0, std = self.noise_std)
                # Noise
                if self.hyper_param['reduced_model'] == False:
                    z = torch.randn((self.hyper_param['batch_size'], 1024, 8)).to(self.device)
                else:
                    z = torch.randn((self.hyper_param['batch_size'], 1024, 16)).to(self.device)
                # Enhanced
                degraded = self.G(original, z)
                # Real pair and fake pair
                real_pair = self.D(decoded, original)
                fake_pair = self.D(degraded, original)

                self.D_opt.zero_grad()
                
                if self.hyper_param['gradient_penalty'] == True:
                    # Gradient panelty
                    alpha = torch.rand(self.hyper_param['batch_size'], 1, 1).to(self.device)
                    interpolated = alpha * decoded + (1 - alpha) * degraded
                    interpolated = interpolated.to(self.device)
                    interpolated.requires_grad_(True)
                    
                    d_interpolated = self.D(interpolated, original)
                    grad_penalty = autograd.grad(outputs = d_interpolated,
                                            inputs = interpolated,
                                            grad_outputs = label_real,
                                            create_graph = True,
                                            retain_graph = True,
                                            only_inputs = True
                                            )[0]
                    grad_penalty = self.hyper_param['lambda_gp'] * ((grad_penalty.norm(2, 1) - 1) ** 2).mean()

                # Discriminator loss
                D_real_loss = self.MSE_loss(real_pair, label_real) # D_real_loss = 0.5 * torch.mean((real_pair - 1.0) ** 2)
                D_fake_loss = self.MSE_loss(fake_pair, label_fake) # D_fake_loss = 0.5 * torch.mean(fake_pair ** 2)
                
                if self.hyper_param['gradient_penalty'] == True:
                    D_loss = 0.5 * (D_real_loss + D_fake_loss) + grad_penalty
                else:
                    D_loss = 0.5 * (D_real_loss + D_fake_loss)
                
                D_loss.backward()
                self.D_opt.step()

                epoch_hist['D_real_loss'].append(D_real_loss.item()) # tensor.item: returns the value of this tensor as a python data.
                epoch_hist['D_fake_loss'].append(D_fake_loss.item())
                
                # Generator loss
                self.G_opt.zero_grad()

                degraded = self.G(original, z)
                fake_pair = self.D(degraded, original)
                
                G_GAN_loss = self.MSE_loss(fake_pair, label_real) # G_GAN_loss = 0.5 * torch.mean((fake_pair - 1.0) ** 2)
                G_L1_loss = self.L1_loss(degraded, decoded)
                
                G_loss = 0.5 * G_GAN_loss + epoch_hyper_param['lambda_G'] * G_L1_loss
                G_loss.backward()
                self.G_opt.step()
                
                epoch_hist['G_GAN_loss'].append(G_GAN_loss.item())
                epoch_hist['G_L1_loss'].append(G_L1_loss.item())
                    
                if np.any(self.info_interval == (iteration + 1)) == True:
                    print("[{:4d}/{:4d} ({:3d}%)]: D_loss: {:.4f}, D_r_loss: {:.4f}, D_f_loss: {:.4f}, G_loss: {:.4f}, G_GAN_loss: {:.4f}, G_L1_loss: {:.4f}, D(o, d): {:.3f}, D(G(d, z), d): {:.3f}"
                          .format(iteration + 1,
                                  self.iteration_num,
                                  ((iteration + 1)*100 // self.iteration_num),
                                  D_loss.item(),
                                  D_real_loss.item(),
                                  D_fake_loss.item(),
                                  G_loss.item(),
                                  G_GAN_loss.item(),
                                  G_L1_loss.item(),
                                  real_pair.mean().item(),
                                  fake_pair.mean().item()))
                    
            # record the train histories
            self.train_hist_iter['D_real_loss'].extend(epoch_hist['D_real_loss']) 
            self.train_hist_iter['D_fake_loss'].extend(epoch_hist['D_fake_loss']) 
            self.train_hist_iter['G_GAN_loss'].extend(epoch_hist['G_GAN_loss'])
            self.train_hist_iter['G_L1_loss'].extend(epoch_hist['G_L1_loss'])
            
            self.train_hist_epoch['D_real_loss'].append(np.array(epoch_hist['D_real_loss']).mean()) # np.array for calculating mean value
            self.train_hist_epoch['D_fake_loss'].append(np.array(epoch_hist['D_fake_loss']).mean())
            self.train_hist_epoch['G_GAN_loss'].append(np.array(epoch_hist['G_GAN_loss']).mean())
            self.train_hist_epoch['G_L1_loss'].append(np.array(epoch_hist['G_L1_loss']).mean())
            self.train_hist_epoch['per_epoch_time'].append(time.time() - epoch_start_time)
            
            avg_epoch_time = int(np.mean(self.train_hist_epoch['per_epoch_time']))
            print("  Avg. 1 epoch time: [%s] / Est. remaining time: [%s]" % (str(datetime.timedelta(seconds = avg_epoch_time)),
                                                                                        str(datetime.timedelta(seconds = (self.hyper_param['epochs'] - epoch - 1)*avg_epoch_time))))
            
            if (epoch + 1) % self.opt['save_epoch_interval'] == 0:
                self.save(epoch)
            
            # learning rate scheduler
            if self.G_scheduler != None:
                self.G_scheduler.step()
            if self.D_scheduler != None:
                self.D_scheduler.step()
                
            self.inference(epoch = epoch + 1,
                           input_file = os.path.join(self.dir['inf_sample'], 'original.wav'),
                           output_file = os.path.join(self.dir['inference'], 'degraded_epoch_{}.wav'.format(epoch + 1))
                           )
            
            # plot losses
            plot_loss(hist = self.train_hist_epoch, plot_dir = self.dir['plot'], x_label = 'Epochs')
            plot_loss(hist = self.train_hist_iter, plot_dir = self.dir['plot'], x_label = 'Iterations')

                
        print('Training Finish.')

       
    def save(self, epoch):
        generator_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_Generator' + '_epoch_' + str(epoch + 1) + '.pth'
        discriminator_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_Discriminator' + '_epoch_' + str(epoch + 1) + '.pth'
        hist_epoch_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_epoch_' + str(epoch + 1) + '_history_epoch.pkl'
        hist_iter_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_epoch_' + str(epoch + 1) + '_history_iter.pkl'
        hyper_param_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_epoch_' + str(epoch + 1) + '_hyper_param.pkl'
        
        if len(self.device_idx) > 1:
            torch.save(
                    obj = {'epoch': epoch,
                           'G_state_dict': self.G.module.state_dict(),
                           'G_opt_state_dict': self.G_opt.state_dict()
                           },
                    f = generator_file,
                    pickle_protocol = 4
                    )
            torch.save(
                    obj = {'epoch': epoch,
                           'D_state_dict': self.D.module.state_dict(),
                           'D_opt_state_dict': self.D_opt.state_dict()
                           },
                    f = discriminator_file,
                    pickle_protocol = 4
                    )
            
        else:
            torch.save(
                    obj = {'epoch': epoch,
                           'G_state_dict': self.G.state_dict(),
                           'G_opt_state_dict': self.G_opt.state_dict()
                           },
                    f = generator_file,
                    pickle_protocol = 4
                    )
            torch.save(
                    obj = {'epoch': epoch,
                           'D_state_dict': self.D.state_dict(),
                           'D_opt_state_dict': self.D_opt.state_dict()
                           },
                    f = discriminator_file,
                    pickle_protocol = 4
                    )
        
        with open(hyper_param_file, 'wb') as f:
            pickle.dump(self.hyper_param, f)
        with open(hist_epoch_file, 'wb') as f:
            pickle.dump(self.train_hist_epoch, f)
        with open(hist_iter_file, 'wb') as f:
            pickle.dump(self.train_hist_iter, f)
             
        print('  epoch:', epoch + 1, ', model saved!')

    def load_generator(self, epoch, map_loc):
        generator_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_Generator' + '_epoch_' + str(epoch) + '.pth'
        if epoch <= 0:
            self.opt['loaded_epoch'] = 0
            return
        if os.path.isfile(generator_file):
            checkpoint = torch.load(generator_file, map_location = map_loc)
            self.G.load_state_dict(checkpoint['G_state_dict'])
            if self.opt['train'] == True:
                self.G_opt.load_state_dict(checkpoint['G_opt_state_dict'])
            self.opt['loaded_epoch'] = checkpoint['epoch'] + 1
        else:
            print('no generator_checkpoint found at {}'.format(self.dir['model']))
            return
    
    def load_discriminator(self, epoch, map_loc):
        discriminator_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_Discriminator' + '_epoch_' + str(epoch) + '.pth'
        if epoch <= 0:
            self.opt['loaded_epoch'] = 0
            return
        if os.path.isfile(discriminator_file):
            checkpoint = torch.load(discriminator_file, map_location = map_loc)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.D_opt.load_state_dict(checkpoint['D_opt_state_dict'])
            self.opt['loaded_epoch'] = checkpoint['epoch'] + 1
        else:
            print('no discriminator_checkpoint found at {}'.format(self.dir['model']))
            return
    
    def load_hist(self, epoch):
        if epoch <= 0:
            self.opt['loaded_epoch'] = 0
            return
        hist_epoch_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_epoch_' + str(epoch) + '_history_epoch.pkl'
        hist_iter_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_epoch_' + str(epoch) + '_history_iter.pkl'
        with open(hist_epoch_file, 'rb') as f:
            self.train_hist_epoch = pickle.load(f)
        with open(hist_iter_file, 'rb') as f:
            self.train_hist_iter = pickle.load(f)
            
    def load_hyper_param(self, epoch):
        if epoch <= 0:
            self.opt['loaded_epoch'] = 0
            return
        
        hyper_param_file = os.path.join(self.dir['model'], self.opt['model_name']) + '_epoch_' + str(epoch) + '_hyper_param.pkl'
        with open(hyper_param_file, 'rb') as f:
            self.hyper_param = pickle.load(f)
            
    def inference(self, epoch, input_file, output_file = None):   #path of wav
        self.G.eval()
        directory = os.path.split(input_file)[0]
        file_name = (os.path.split(input_file)[1]).split('.')[0]
        
        wav, _ = lbr.load(input_file, sr = self.opt['sampling_rate'], dtype = np.float32)
        
        length_original = len(wav)
        
        wavs = np.array(extract_chunk(wav = wav, overlap = 1)) # extract chunks and transform np.array
        wavs = wavs.reshape(wavs.shape[0], 1, wavs.shape[1]) # (chunk_num, 16384) -> (chunk_num, 1, 16384)
        wavs = pre_emphasis(wavs)
        wavs = torch.from_numpy(wavs).to(device = self.device)
        if self.hyper_param['reduced_model'] == False:
            z = torch.randn((wavs.shape[0], 1024, 8)).to(device = self.device)
        else:
            z = torch.randn((wavs.shape[0], 1024, 16)).to(device = self.device)
        
        transformed_wavs = self.G(wavs, z)
        
        if self.device == 'cpu':
            transformed_wavs = transformed_wavs.detach().numpy()    
        else:
            transformed_wavs = transformed_wavs.detach().cpu().numpy()
        transformed_wavs = de_emphasis(transformed_wavs)
        
        transformed_wav = transformed_wavs.reshape(-1)
        transformed_wav = transformed_wav[: length_original]
        
        if output_file == None:
            lbr.output.write_wav(os.path.join(directory, file_name + '_transformd_epoch_{}.wav'.format(epoch)), transformed_wav, self.opt['sampling_rate'])
        else:
            lbr.output.write_wav(output_file, transformed_wav, self.opt['sampling_rate'])
        
    def test(self, path = os.path.join(os.path.abspath(''), 'raw_dataset', 'clean_testset_wav_decoded')):
        path_decoded = path
        files = os.listdir(path_decoded)
        
        for file in files:
            file = os.path.join(path_decoded, file)
            directory = os.path.split(file)[0]
            file_name = (os.path.split(file)[1]).split('.')[0]
            if (os.path.split(file)[1]).split('.')[1] != 'wav':
                continue
            
            wav, _ = lbr.load(file, sr = self.opt['sampling_rate'], dtype = np.float32)
            length_original = len(wav)

            wavs = np.array(extract_chunk(wav = wav, overlap = 1)) # extract chunks and transform np.array
            wavs = wavs.reshape(wavs.shape[0], 1, wavs.shape[1]) # (chunk_num, 16384) -> (chunk_num, 1, 16384)
            wavs = pre_emphasis(wavs)
            wavs = torch.from_numpy(wavs).to(device = self.device)
            if self.hyper_param['reduced_model'] == False:
                z = torch.randn((wavs.shape[0], 1024, 8)).to(device = self.device)
            else:
                z = torch.randn((wavs.shape[0], 1024, 16)).to(device = self.device)
            
            transformed_wavs = self.G(wavs, z)
            
            if self.device == 'cpu':
                transformed_wavs = transformed_wavs.detach().numpy()    
            else:
                transformed_wavs = transformed_wavs.detach().cpu().numpy()
            transformed_wavs = de_emphasis(transformed_wavs)
            
            transformed_wav = transformed_wavs.reshape(-1)
            transformed_wav = transformed_wav[: length_original]

            output_path = os.path.join(os.path.abspath(''), 'test_results', os.path.split(file)[1])
            lbr.output.write_wav(output_path, transformed_wav, self.opt['sampling_rate'])       
            