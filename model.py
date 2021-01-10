import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
from torch import tan, atan
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
BATCH_SIZE = 4#3#
INPUT_AXIS = 784
LATENT_DIMENSION = 2
shape_log = False
A = int(BATCH_SIZE*INPUT_AXIS)#1632
B = int(BATCH_SIZE*(INPUT_AXIS/2))
C = int(BATCH_SIZE*(INPUT_AXIS/4))
D = int(BATCH_SIZE*2)#int(C/4)batch_size*(axis-1)
E = int(BATCH_SIZE*LATENT_DIMENSION)#int(C/4)batch_size*(axis-1)

##########外側5%を省いて学習を繰り返す
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),###############畳み込みを粗くする
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
            )

        self.full_connection = nn.Sequential(
            nn.Linear(BATCH_SIZE*64, BATCH_SIZE*LATENT_DIMENSION),
            )

        self.tr_full_connection = nn.Sequential(
            nn.Linear(BATCH_SIZE*LATENT_DIMENSION, BATCH_SIZE*64),
            )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),

            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=0),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, 3, padding=1),
            #nn.Dropout(0.1),
            nn.Sigmoid(),
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
        in_min = torch.min(in_data).cuda()
        in_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):#
            in_cord1 = in_min + in_data[n[0]]
            in_cord2 = in_min + in_data[n[1]]
            #print(n, '|', in_cord1, '|', in_cord2)
            in_diff_list.append(torch.sqrt(torch.sum((in_cord1-in_cord2)**2)))#L2ノルムでいいのか？
        in_diff_sum = torch.stack(in_diff_list, dim=0).cuda()

        lat_min  = torch.min(lat_repr).cuda()
        lat_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):#
            lat_cord1 = lat_min + lat_repr[n[0]]
            lat_cord2 = lat_min + lat_repr[n[1]]
            #print(n, '|', lat_cord1, '|', lat_cord2)
            lat_diff_list.append(torch.sqrt(torch.sum((lat_cord1-lat_cord2)**2)))
        lat_diff_sum = torch.stack(lat_diff_list, dim=0).cuda()

        
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{z.shape}')
        y = y.view(BATCH_SIZE, 64, 1, 1)
        output = self.decoder(y).cuda()
        if shape_log == True: print(f'y:{output.shape}')
        #-----------------------------------------

        out_min = torch.min(output).cuda()
        out_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):#
            out_cord1 = out_min + output[n[0]]
            out_cord2 = out_min + output[n[1]]
            out_diff_list.append(torch.sqrt(torch.sum((out_cord1-out_cord2)**2)))#L2ノルムでいいのか？
        out_diff_sum = torch.stack(out_diff_list, dim=0).cuda()

        return output, in_diff_sum, lat_diff_sum, out_diff_sum, lat_repr

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = 1
        self.dim_h = 1
        self.n_z = 1

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()#出力は0~1でサイズは[batch_size, 1]
        )

    def forward(self, x):
        x = self.main(x)
        return x