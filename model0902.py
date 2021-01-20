import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
from torch import tan, atan
BATCH_SIZE = 3#3#
INPUT_AXIS = 3
LATENT_DIMENSION = 2
shape_log = False

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(BATCH_SIZE*INPUT_AXIS, BATCH_SIZE*INPUT_AXIS),####
            )
        self.full_connection = nn.Sequential(
            nn.Linear(BATCH_SIZE*INPUT_AXIS, BATCH_SIZE*LATENT_DIMENSION),####
            # nn.Sigmoid()
        )
        self.tr_full_connection = nn.Sequential(
            nn.Linear(BATCH_SIZE*LATENT_DIMENSION, BATCH_SIZE*INPUT_AXIS),####
        )
        self.decoder = nn.Sequential(
            nn.Linear(BATCH_SIZE*INPUT_AXIS, BATCH_SIZE*INPUT_AXIS),####
            nn.Sigmoid(),####Tanh
            )
    
    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        z = self.encoder(x).cuda()
        if shape_log == True: print(f'z = encorder(x):{z.shape}')
        z = z.view(-1)
        z = self.full_connection(z).cuda()
        lat_repr = z.reshape(BATCH_SIZE, LATENT_DIMENSION).cuda()#################################
        if shape_log == True: print(f'full_connection(z):{z.shape}')

        #----------------------------------------
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS).cuda()

        #print(lat_repr)
        in_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):#
            in_cord1 = 1 + in_data[n[0]]
            in_cord2 = 1 + in_data[n[1]]
            #print(n, '|', in_cord1, '|', in_cord2)
            in_diff_list.append(torch.sqrt(torch.sum((in_cord1-in_cord2)**2))**2)#L2ノルムでいいのか？
        in_diff_sum = torch.stack(in_diff_list, dim=0).cuda()

        lat_min  = torch.min(lat_repr).cuda()
        lat_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):#
            lat_cord1 = 1 + lat_repr[n[0]]
            lat_cord2 = 1 + lat_repr[n[1]]
            #print(n, '|', lat_cord1, '|', lat_cord2)
            lat_diff_list.append(torch.sqrt(torch.sum((lat_cord1-lat_cord2)**2))**2)
        lat_diff_sum = torch.stack(lat_diff_list, dim=0).cuda()

        
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{z.shape}')
        y = y.view(BATCH_SIZE*INPUT_AXIS)
        output = self.decoder(y).cuda()
        if shape_log == True: print(f'y:{output.shape}')
        #-----------------------------------------

        out_min = torch.min(output).cuda()
        out_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):#
            out_cord1 = 1 + output[n[0]]
            out_cord2 = 1 + output[n[1]]
            out_diff_list.append(torch.sqrt(torch.sum((out_cord1-out_cord2)**2))**2)#L2ノルムでいいのか？
        out_diff_sum = torch.stack(out_diff_list, dim=0).cuda()

        return output, in_diff_sum, lat_diff_sum, out_diff_sum, lat_repr
