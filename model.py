import torch
import torch.nn as nn
from util import weights_init, weights_init_xavier
from torch.nn import utils
# Shape notation: [Batch x Channel x Length]
# Each layer equals a function. So I writed down inputs and outputs dimensions to each layers.

class Generator(nn.Module):

    def __init__(self, reduced = False):
        super(Generator, self).__init__()
            
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        if reduced == False:
            self.conv_channels = [(1,16), (16, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 512), (512, 1024)]
            self.conv_params = {'kernel_size': 31,
                                'stride': 2,
                                'padding': 15,
                                'dilation': 1,
                                'groups': 1,
                                'bias': False
                                }
        else:
            self.conv_channels = [(1,64), (64, 128), (128, 256), (256, 512), (512, 1024)]
            self.conv_params = {'kernel_size': 31,
                                'stride': 4,
                                'padding': 15,
                                'dilation': 1,
                                'groups': 1,
                                'bias': False
                                }

        for layer_idx in range(len(self.conv_channels)):
            self.conv_params['in_channels'] = self.conv_channels[layer_idx][0]
            self.conv_params['out_channels'] = self.conv_channels[layer_idx][1]
            modules = [nn.Conv1d(**self.conv_params),
                       nn.PReLU()
                       ]
            self.enc.append(nn.Sequential(*modules))

        if reduced == False:
            self.conv_params['output_padding'] = 1
        else:
            self.conv_params['output_padding'] = 3
            
        for layer_idx in range(len(self.conv_channels)):
            self.conv_params['in_channels'] = 2 * self.conv_channels[layer_idx][1]
            self.conv_params['out_channels'] = self.conv_channels[layer_idx][0]
            
            modules = [nn.ConvTranspose1d(**self.conv_params),
                       nn.PReLU() if layer_idx != 0 else nn.Tanh()
                       ]
            self.dec.append(nn.Sequential(*modules))
        
        
        weights_init(self)
        
        """
        Encoder Part:
            self.enc[0] = [B x 1 x 16384]->[B x 16 x 8192]
            self.enc[1] = [B x 16 x 8192]->[B x 32 x 4096]
            self.enc[2] = [B x 32 x 4096]->[B x 32 x 2048]
            self.enc[3] = [B x 32 x 2048]->[B x 64 x 1024]
            self.enc[4] = [B x 64 x 1024]->[B x 64 x 512]
            self.enc[5] = [B x 64 x 512]->[B x 128 x 256]
            self.enc[6] = [B x 128 x 256]->[B x 128 x 128]
            self.enc[7] = [B x 128 x 128]->[B x 256 x 64]
            self.enc[8] = [B x 256 x 64]->[B x 256 x 32]
            self.enc[9] = [B x 256 x 32]->[B x 512 x 16]
            self.enc[10] = [B x 512 x 16]->[B x 1024 x 8]
        Decoder Part:
            self.dec[10] = [B x 1024+1024 x 8]->[B x 512 x 16]
            self.dec[9] = [B x 512+512 x 16]->[B x 256 x 32]
            self.dec[8] = [B x 256+256 x 32]->[B x 256 x 64]
            self.dec[7] = [B x 256+256 x 64]->[B x 128 x 128]
            self.dec[6] = [B x 128+128 x 128]->[B x 128 x 256]
            self.dec[5] = [B x 128+128 x 256]->[B x 64 x 512]
            self.dec[4] = [B x 64+64 x 512]->[B x 64 x 1024]
            self.dec[3] = [B x 64+64 x 1024]->[B x 32 x 2048]
            self.dec[2] = [B x 32+32 x 2048]->[B x 32 x 4096]
            self.dec[1] = [B x 32+32 x 4096]->[B x 16 x 8192]
            self.dec[0] = [B x 16+16 x 8192]->[B x 1 x 16384]
        """

    def forward(self, x, z):
        # x: noisy signals
        # z: latent vectors

        # Encoder part: Input is a noisy signal(mono channel) as shape [B x 1 x 16384].
        e = []
        for layer_idx in range(len(self.conv_channels)):
            if layer_idx > 0:
                e.append(self.enc[layer_idx](e[layer_idx - 1]))
            else:
                e.append(self.enc[layer_idx](x))

        # Decoder part: Input is a concatenated tensor (Latent Vector z, Thought Vector c)   
        output = z
        
        for layer_idx in range(len(self.conv_channels) - 1, -1, -1):
            output = torch.cat((output, e[layer_idx]), dim = 1) # dim = 1; Channel Concatenate (previous layer's output, homologous encoded data)
            output = self.dec[layer_idx](output)

        return output

# Discriminator
class Discriminator(nn.Module):

    def __init__(self, reduced = False, spectral_norm = False, batch_norm = True):
        super(Discriminator, self).__init__()
        
        if reduced == False:
            self.conv_channels = [(2, 32), (32, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 512), (512, 512), (512, 1024), (1024, 2048)]
            self.conv_params = {'kernel_size': 31,
                                'stride': 2,
                                'padding': 15,
                                'dilation': 1,
                                'groups': 1,
                                'bias': False
                                }
        else:
            self.conv_channels = [(2, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
            self.conv_params = {'kernel_size': 31,
                                'stride': 4,
                                'padding': 15,
                                'dilation': 1,
                                'groups': 1,
                                'bias': False
                                }
        
        self.Conv_layers = []
        """
        Conv_layers:
            [B x 2 x 16384]->[B x 16 x 8192]
            [B x 16 x 8192]->[B x 32 x 4096]
            [B x 32 x 4096]->[B x 32 x 2048]
            [B x 32 x 2048]->[B x 64 x 1024]
            [B x 64 x 1024]->[B x 64 x 512]
            [B x 64 x 512]->[B x 128 x 256]
            [B x 128 x 256]->[B x 128 x 128]
            [B x 128 x 128]->[B x 256 x 64]
            [B x 256 x 64]->[B x 256 x 32]
            [B x 256 x 32]->[B x 512 x 16]
            [B x 512 x 16]->[B x 1024 x 8]
        """
        for layer_idx in range(len(self.conv_channels)):
            self.conv_params['in_channels'] = self.conv_channels[layer_idx][0]
            self.conv_params['out_channels'] = self.conv_channels[layer_idx][1]
            if batch_norm == True:
                modules = [nn.Conv1d(**self.conv_params),
                           nn.BatchNorm1d(self.conv_params['out_channels']),
                           nn.PReLU(),
                           nn.Dropout()
                           ]
            else:
                modules = [nn.Conv1d(**self.conv_params),
                           nn.PReLU(),
                           nn.Dropout()
                           ]                
#            if (layer_idx + 1) % 3 == 0:
#                modules.insert(1, nn.Dropout())
            self.Conv_layers.extend(modules)
            
        self.Conv_layers = nn.Sequential(*self.Conv_layers)
        self.Linear_layers = nn.Sequential(nn.Linear(16384, 256),
                                           nn.PReLU(),
                                           nn.Linear(256, 128),
                                           nn.PReLU(),
                                           nn.Linear(128, 1)
                                           )
        weights_init(self)
        
        if spectral_norm == True:
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    m = utils.spectral_norm(m)

    
    
    def forward(self, original, decoded):
        x = torch.cat([original, decoded], 1)  # Channel concatenation
        x = self.Conv_layers(x)
        x = x.view(x.shape[0], -1) # [B x 1 x 8]->[B x 8]
        x = self.Linear_layers(x)

        return x
