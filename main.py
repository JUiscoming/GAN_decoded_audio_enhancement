from core import Core
from model import Generator, Discriminator
import os
# if you execute this code on cpu, 'device_id': -1

def main():
    kwargs = {'model_name': 'LSGAN_SE',
              'reduced_model': False,
              'loaded_epoch': 220,
              'epochs': 250,
              'batch_size': 64,
              'lambda_G': 100,
              'gradient_penalty': False,
              'spectral_normalization': True,
              'D_batch_normalization': False,
              'lambda_gp': 10,
              'device_id': [0],
              'clean_only': True,
              'save_epoch_interval': 2,
              'info_num_per_epoch': 8,
              'G_lr': 5e-5,
              'D_lr': 5e-5,
              'G_lr_scheduler': False,
              'D_lr_scheduler': False,
              'G_lr_cos_scheduler': False,
              'D_lr_cos_scheduler': False,
              'G_lr_cos_scheduler_opt': {'restart_interval': 20, 'shift': True},
              'D_lr_cos_scheduler_opt': {'restart_interval': 20, 'shift': True},
              'G_lr_scheduler_opt': {'step_size': 30, 'gamma': 0.1, 'last_epoch': -1},
              'D_lr_scheduler_opt': {'step_size': 30, 'gamma': 0.1, 'last_epoch': -1},
              'train': True,
              'seed': 113,
              'sampling_rate': 16000
              }
    
    Model = Core(Generator, Discriminator, **kwargs)
#    Model.inference(kwargs['loaded_epoch'], 'decoded_007.wav', 'enhanced_007.wav')
    # Model.train_codec_enhancement()
    # Model.test()    
    Model.test(path = os.path.join(os.path.abspath(''), 'test'))
#    Model.train_codec_approximate()

if __name__ == '__main__':
    main()
