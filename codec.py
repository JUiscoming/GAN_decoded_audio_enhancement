# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:58:23 2020

@author: MSPL
"""

import os
import subprocess

path_main = os.path.abspath('')
path_codec = os.path.join(path_main, 'codec')
path_raw_dataset = os.path.join(path_main, 'raw_dataset')
batch_file = os.path.join(path_codec, 'AAC.bat')
clean_dataset = ['clean_trainset_wav', 'clean_testset_wav']
noisy_dataset = ['noisy_trainset_wav', 'noisy_testset_wav']
dataset = clean_dataset + noisy_dataset

for original in dataset:
    path_original = os.path.join(path_raw_dataset, original)
    path_encoded = path_original + '_encoded'
    path_decoded = path_original + '_decoded'
    
    if not os.path.exists(path_encoded):
        os.makedirs(path_encoded)
    if not os.path.exists(path_decoded):
        os.makedirs(path_decoded)
        
    files = os.listdir(path_original)
    for idx, file in enumerate(files):
        wav_original = os.path.join(path_original, file)
        wav_encoded = os.path.splitext(os.path.join(path_encoded, file))[0] + '.m4a'
        wav_decoded = os.path.join(path_decoded, file)
        
        
        subprocess.run([batch_file,
                        wav_original,
                        wav_encoded,
                        wav_decoded
                        ], shell = True)
        
        print('\r{}/{} files in {} are processed using AAC codec.'.format(idx, len(files), original), end='')
        
    print('processing {} is complete.'.format(original))

print('processing is complete.')