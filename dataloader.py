# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:28:46 2020

@author: MSPL
"""
#from torchvision import datasets, transforms
from torch.utils.data import Dataset
from util import extract_chunk
import numpy as np
import os
import librosa as lbr
import pickle
#import torch
import util
import shutil


###########################################################
#                      Dataset Class                      #
###########################################################
class Valentini_VCTK(Dataset):
    def __init__(self, train = True, pre_emphasis = True, pre_emphasis_coef = 0.95, clean_only = False):
        self.path_main = os.path.abspath('')
        self.path_dataset = os.path.join(self.path_main, 'dataset') # 최종 저장할 곳
        self.path_raw_dataset = os.path.join(self.path_main, 'raw_dataset')
        self.path_clean_original = os.path.join(self.path_raw_dataset, 'clean_{}set_wav'.format('train' if train else 'test'))
        self.path_clean_decoded = os.path.join(self.path_raw_dataset, 'clean_{}set_wav_decoded'.format('train' if train else 'test'))
        self.path_noisy_original = os.path.join(self.path_raw_dataset, 'noisy_{}set_wav'.format('train' if train else 'test'))
        self.path_noisy_decoded = os.path.join(self.path_raw_dataset, 'noisy_{}set_wav_decoded'.format('train' if train else 'test'))
        self.path_original = [self.path_clean_original]
        self.path_decoded = [self.path_clean_decoded]
        if clean_only == False:
            self.path_original.append(self.path_noisy_original)
            self.path_decoded.append(self.path_noisy_decoded)
      
        
        self.original = []  #paths of wav
        self.decoded = []
        
        self.original_np = []
        self.decoded_np = []

        self.pre_emphasis_coef = pre_emphasis_coef
        
        if not os.path.exists(self.path_dataset):
            os.makedirs(self.path_dataset)
        
        condition = [os.path.join(self.path_dataset, '{}_original.npy'.format('train' if train else 'test')),
                     os.path.join(self.path_dataset, '{}_decoded.npy'.format('train' if train else 'test'))]

        condition = list(map(os.path.isfile, condition))
        
        if False in condition:  # dataset을 처음 만드는 경우
            print('first execution: make dataset.')
            for path in self.path_original:
                self.original.extend(self.get_wav_list(path))
            for path in self.path_decoded:
                self.decoded.extend(self.get_wav_list(path))
            self.original.sort()
            self.decoded.sort()
        # self.original에는 original clean, noisy의 path가, self.decoded에는 decoded clean, noisy의 path가 존재.
        
            self.original, _ = zip(*map(self.wav_load_16000hz, self.original))  #y, sr = lbr.load(audio_path), extract y
            self.decoded, _ = zip(*map(self.wav_load_16000hz, self.decoded))         
        
            for (wav_original, wav_decoded) in list(zip(self.original, self.decoded)):
                # 1. Align length
                wav_decoded = self.align_length(align_target = wav_decoded, length = len(wav_original))
                # 2. Extract chunks
                self.original_np.extend(extract_chunk(wav = wav_original, overlap = 0.5))
                self.decoded_np.extend(extract_chunk(wav = wav_decoded, overlap = 0.5))
        
            del(self.original)
            del(self.decoded)
            
            self.original_np = np.array(self.original_np, dtype=np.float32)
            self.decoded_np = np.array(self.decoded_np, dtype=np.float32)
            
            self.original_np = np.expand_dims(self.original_np, axis=1)   #[batch, length] -> [batch, channel=1, length]
            self.decoded_np = np.expand_dims(self.decoded_np, axis=1)
            
            # pre-emphasis
            if pre_emphasis == True:
                self.original_np = util.pre_emphasis(self.original_np)
                self.decoded_np = util.pre_emphasis(self.decoded_np)
        
            np.save(os.path.join(self.path_dataset, '{}_original'.format('train' if train else 'test')), self.original_np)
            np.save(os.path.join(self.path_dataset, '{}_decoded'.format('train' if train else 'test')), self.decoded_np)
            
        self.original_np = np.load(os.path.join(self.path_dataset, '{}_original.npy'.format('train' if train else 'test')))
        self.decoded_np = np.load(os.path.join(self.path_dataset, '{}_decoded.npy'.format('train' if train else 'test')))

    def __len__(self):
        #length of dataset (=# of samples)
        return len(self.original_np)

    def __getitem__(self, idx):
        #function of get a sample in the dataset
        return self.original_np[idx], self.decoded_np[idx]
            
    def get_wav_list(self, folder):
        wav_list = []
        files = os.listdir(folder)
        for file in files:
            path_full = os.path.join(folder, file)
            _, ext = os.path.splitext(path_full)
            if ext == '.wav':
                wav_list.append(path_full)
    
        return wav_list                
    
    def wav_load_16000hz(self, path): # it is made to use map function input.
        return lbr.load(path, sr = 16000, dtype = np.float32)
    
    def align_length(self, align_target, length):   # align_object.dtype = align_target.dtype = np.array
        diff_len = length - len(align_target)
        if diff_len > 0:
            align_target = np.r_[align_target, np.array([0]*diff_len, dtype = np.float32)] # np.r_[a,b] == np.concatenate((a,b), axis = 0)
        elif diff_len < 0:
            align_target = align_target[:diff_len]
        return align_target
    
#                
#class VoicebankDataset(Dataset):
#    def __init__(self, train = True, pre_emphasis = True):
#        self.train = train
#        #pre-processing 
#        
#        self.train_original = []
#        self.test_original = []
#        self.train_decoded = []
#        self.test_decoded = []
#        self.pre_emphasis_alpha = 0.95
#        self.data_set_num = 30
#        self.train_set_num = 28
#        
#        
#        if not os.path.exists(path_dataset):
#            os.makedirs(path_dataset)
#        
#        condition = [os.path.join(path_dataset, 'train_original.npy'),
#                     os.path.join(path_dataset, 'test_original.npy'),
#                     os.path.join(path_dataset, 'train_decoded.npy'),
#                     os.path.join(path_dataset, 'test_decoded.npy')]
#        condition = list(map(os.path.isfile, condition))
#        if False in condition:  # dataset을 처음 만드는 경우
#            ###########################################################
#            #                  1. Training set waveform               #
#            ###########################################################   
#            person_list = get_person_list(path_original)
#            
#            with open(os.path.join(path_dataset, 'person_gender.pkl'), 'rb') as f:
#                person_gender = pickle.load(f)
#            male_list = []
#            female_list = []
#            
#            for person, gender in zip(person_list, person_gender):
#                if gender == 'F':
#                    female_list.append(person)
#                elif gender == 'M':
#                    male_list.append(person)
#
#            train_male_list, test_male_list = sampling_in_person(male_list, self.data_set_num//2, self.train_set_num//2)
#            train_female_list, test_female_list = sampling_in_person(female_list, self.data_set_num//2, self.train_set_num//2)
#                   
#            train_person_list, test_person_list = [*train_male_list, *train_female_list], [*test_male_list, *test_female_list]
#            train_person_list.sort()
#            test_person_list.sort()
#           
#            with open(os.path.join(path_dataset, 'person_list.txt'), 'w') as f:
#                f.write('train_person_list\nMale: {}\nFemale: {}\ntest_person_list\nMale: {}\nFemale: {}\n'.format(train_male_list, train_female_list, test_male_list, test_female_list))
#            
#            for i, person in enumerate(train_person_list):
#                path_original_person = os.path.join(path_original, person)
#                path_decoded_person = os.path.join(path_decoded, person)
#                wav_original_list = dir_wav(path_original_person)
#                wav_decoded_list = dir_wav(path_decoded_person)
#                wav_original_list, _ = zip(*map(wav_load_16000hz, wav_original_list))  #y, sr = lbr.load(audio_path)
#                wav_decoded_list, _ = zip(*map(wav_load_16000hz, wav_decoded_list))              
#                             
#                for (wav_original, wav_decoded) in list(zip(wav_original_list, wav_decoded_list)):
#                    # 1. Align length
#                    wav_decoded = align_length(align_target = wav_decoded, length = len(wav_original))
#                    # 2. Extract chunks
#                    self.train_original.extend(extract_chunk(wav_original))
#                    self.train_decoded.extend(extract_chunk(wav_decoded))
#                print('person {}/{}: training set waveforms are segmented.'.format(i+1, len(train_person_list)))
#
#            ###########################################################
#            #                  2. Test set waveform                   #
#            ###########################################################
#            for i, person in enumerate(test_person_list):
#                path_original_person = os.path.join(path_original, person)
#                path_decoded_person = os.path.join(path_decoded, person)
#                wav_original_list = dir_wav(path_original_person)
#                wav_decoded_list = dir_wav(path_decoded_person)
#                wav_original_list, _ = zip(*map(wav_load_16000hz, wav_original_list))  #y, sr = lbr.load(audio_path)
#                wav_decoded_list, _ = zip(*map(wav_load_16000hz, wav_decoded_list))
#                
#                for (wav_original, wav_decoded) in list(zip(wav_original_list, wav_decoded_list)):
#                    # 1. Align length
#                    wav_decoded = align_length(align_target = wav_decoded, length = len(wav_original))             
#                    # 2. Extract chunks
#                    self.test_original.extend(extract_chunk(wav_original))
#                    self.test_decoded.extend(extract_chunk(wav_decoded))
#                print('person {}/{}: test set waveforms are segmented.'.format(i+1, len(test_person_list)))
#
#            ###########################################################
#            #                  3. List to numpy array                 #
#            ###########################################################
#            self.train_original = np.array(self.train_original, dtype=np.float32)
#            self.train_decoded = np.array(self.train_decoded, dtype=np.float32)
#            self.test_original = np.array(self.test_original, dtype=np.float32)
#            self.test_decoded = np.array(self.test_decoded, dtype=np.float32)
#            
#            self.train_original = np.expand_dims(self.train_original, axis=1)   #[batch, length] -> [batch, channel=1, length]
#            self.train_decoded = np.expand_dims(self.train_decoded, axis=1)
#            self.test_original = np.expand_dims(self.test_original, axis=1)
#            self.test_decoded = np.expand_dims(self.test_decoded, axis=1)
#            
#            # pre-emphasis
#            if pre_emphasis == True:
#                self.train_original = util.pre_emphasis(self.train_original)
#                self.train_decoded = util.pre_emphasis(self.train_decoded)
#                self.test_original = util.pre_emphasis(self.test_original)
#                self.test_decoded = util.pre_emphasis(self.test_decoded)
#
#            np.save(os.path.join(path_dataset, 'train_original'), self.train_original)
#            np.save(os.path.join(path_dataset, 'train_decoded'), self.train_decoded)
#            np.save(os.path.join(path_dataset, 'test_original'), self.test_original)
#            np.save(os.path.join(path_dataset, 'test_decoded'), self.test_decoded)
#
#        self.train_original = np.load(os.path.join(path_dataset, 'train_original.npy'))
#        self.train_decoded = np.load(os.path.join(path_dataset, 'train_decoded.npy'))
#        self.test_original = np.load(os.path.join(path_dataset, 'test_original.npy'))
#        self.test_decoded = np.load(os.path.join(path_dataset, 'test_decoded.npy'))
#            
#            
#    def __len__(self):
#        #length of dataset (=# of samples)
#        if self.train:
#            return len(self.train_original)
#        else:
#            return len(self.test_original)
#        
#    def __getitem__(self, idx):
#        #function of get a sample in the dataset
#        if self.train:
#            return self.train_original[idx], self.train_decoded[idx]
#        else:
#            return self.test_original[idx], self.test_decoded[idx]
