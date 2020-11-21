import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
from torch import tan, atan
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
BATCH_SIZE = 3#3#
INPUT_AXIS = 784
LATENT_DIMENSION = 2
shape_log = False
default_Gradient = 0.00001
delete_max = 10
A = int(BATCH_SIZE*INPUT_AXIS)#1536
B = int(BATCH_SIZE*(INPUT_AXIS/2))
C = int(BATCH_SIZE*(INPUT_AXIS/4))
D = int(BATCH_SIZE*2)#int(C/4)batch_size*(axis-1)
E = int(BATCH_SIZE*LATENT_DIMENSION)#int(C/4)batch_size*(axis-1)

##########外側5%を省いて学習を繰り返す
class autoencoder(nn.Module):
    def __init__(self, Gradient = default_Gradient):
        super(autoencoder, self).__init__()
        
        self.Gradient = Parameter(torch.tensor(Gradient)**2)#2乗で正
        self.Gradient.requiresGrad = True

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 15, 3, stride=2, padding=1),###############畳み込みを粗くする
            nn.BatchNorm2d(15),
            nn.ReLU(),

            nn.Conv2d(15, 30, 3, stride=2, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),

            nn.Conv2d(30, 60, 3, stride=2, padding=0),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.Conv2d(60, 60, 3, stride=2, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.Conv2d(60, 60, 3, stride=2, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            )

        self.full_connection = nn.Sequential(
            nn.Linear(BATCH_SIZE*60, BATCH_SIZE*36),
            nn.Dropout(0.1),
            # nn.Linear(BATCH_SIZE*36, BATCH_SIZE*12),
            # nn.Dropout(0.1),
            nn.Linear(BATCH_SIZE*36, BATCH_SIZE*LATENT_DIMENSION),
            )

        self.tr_full_connection = nn.Sequential(
            nn.Linear(BATCH_SIZE*LATENT_DIMENSION, BATCH_SIZE*36),
            nn.Dropout(0.1),
            # nn.Linear(BATCH_SIZE*12, BATCH_SIZE*36),
            # nn.Dropout(0.1),
            nn.Linear(BATCH_SIZE*36, BATCH_SIZE*60),
            )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(60, 30, 3, padding=0),
            nn.BatchNorm2d(30),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(30, 15, 3, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),

            nn.Conv2d(15, 1, 3, padding=1),
            #nn.Dropout(0.1),
            nn.Tanh(),

            #-------------------------------------------
            # nn.ConvTranspose2d(64, 32, 7),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            # nn.Tanh()
            )

        self.Softmax1 = nn.Softmax()
        self.Softmax2 = nn.Softmax()

    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        y = self.encoder(x).cuda()
        if shape_log == True: print(f'y = encorder(x):{y.shape}')
        y = y.view(-1)
        y = self.full_connection(y).cuda()
        lat_repr = y.reshape(BATCH_SIZE, LATENT_DIMENSION).cuda()#################################
        if shape_log == True: print(f'full_connection(y):{y.shape}')

        #----------------------------------------
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS).cuda()

        #print(lat_repr)
        in_min = torch.min(in_data).cuda()
        lat_min  = torch.min(lat_repr).cuda()
        in_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):#同じもの同士を比較しない
            in_cord1 = in_min + in_data[n[0]]
            in_cord2 = in_min + in_data[n[1]]
            #print(n, '|', in_cord1, '|', in_cord2)
            in_diff_list.append(torch.sqrt(torch.sum((in_cord1-in_cord2)**2)))#L2ノルムでいいのか？
        in_diff_sum = torch.stack(in_diff_list, dim=0).cuda()

        lat_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):
            lat_cord1 = lat_min + lat_repr[n[0]]
            lat_cord2 = lat_min + lat_repr[n[1]]
            #print(n, '|', lat_cord1, '|', lat_cord2)
            lat_diff_list.append(torch.sqrt(torch.sum((lat_cord1-lat_cord2)**2)))
        lat_diff_sum = torch.stack(lat_diff_list, dim=0).cuda()

        
        #----------------------------------------
        y = self.tr_full_connection(y).cuda()
        if shape_log == True: print(f'tr_full_connection(y):{y.shape}')
        y = y.view(BATCH_SIZE, 60, 1, 1)
        output = self.decoder(y).cuda()
        if shape_log == True: print(f'decoder(y):{output.shape}')

        ###########大きい距離を無視
        # for x in range(delete_max):
        #     in_diff_sum[torch.argmax(in_diff_sum)] = 0
        #     lat_diff_sum[torch.argmax(lat_diff_sum)] = 0

        #----------------------------------------
        #print(f'{torch.mean(lat_diff_sum)}::::{torch.mean(in_diff_sum)}')
        
        ###########ガウス分布、平均に学習可能なパラメータを用いる
        # exp_in_diff_sum = (1 / torch.sqrt(6.283*torch.var(in_diff_sum))
        # )*torch.exp(-1 * (((in_diff_sum - self.in_mean) **2 ) / (2 * torch.var(in_diff_sum)))).cuda()
        # exp_lat_diff_sum = (1 / torch.sqrt(6.283*torch.var(lat_diff_sum))
        # )*torch.exp(-1 * (((lat_diff_sum - self.lat_mean) **2 ) / (2 * torch.var(lat_diff_sum)))).cuda()
        
        ##########ガウス分布、平均にそのバッチごとの平均を用いる
        # exp_in_diff_sum = (1 / torch.sqrt(6.283*torch.var(in_diff_sum))
        # )*torch.exp(-1 * (((in_diff_sum - torch.mean(in_diff_sum)) **2 ) / (2 * torch.var(in_diff_sum)))).cuda()
        # exp_lat_diff_sum = (1 / torch.sqrt(6.283*torch.var(lat_diff_sum))
        # )*torch.exp(-1 * (((lat_diff_sum - torch.mean(lat_diff_sum)) **2 ) / (2 * torch.var(lat_diff_sum)))).cuda()
        #y = y*self._ReLU1(((-1*self.Gradient)*y)+1)
        #lat_diff_sum = lat_diff_sum*self._ReLU1(((-1*self.Gradient)*lat_diff_sum)+1)
        #in_diff_sum = in_diff_sum*self._ReLU2(((-1*self.Gradient)*in_diff_sum)+1)

        #print(f'{self.lat_mean}____{self.in_mean}')
        # exp_in_diff_sum = self._ReLU1(exp_in_diff_sum)#変換後に距離の値が負になる可能性があるのでReLUにかける
        # exp_lat_diff_sum = self._ReLU2(exp_lat_diff_sum)

        ##############逆さまの正規分布は？逆正弦分布
        ##############手前でロスを取ってから2次元に圧縮(out_repr)
        # lat_diff_sum =  lat_diff_sum - (lat_diff_sum*exp_lat_diff_sum)
        # in_diff_sum = in_diff_sum - (in_diff_sum*exp_in_diff_sum)
        # lat_diff_sum = lat_diff_sum - torch.min(lat_diff_sum)
        # in_diff_sum = in_diff_sum - torch.min(in_diff_sum)
        ##############T分布
        # in_diff_sum = self.Softmax1(in_diff_sum)
        # lat_diff_sum = self.Softmax2(lat_diff_sum)
        perplexity = 1
        exp_in_diff_sum = (1 / (torch.sqrt(torch.Tensor([6.283*perplexity]))))*torch.exp(-1*(((in_diff_sum - 0)**2) / (2*perplexity))).cuda()
        #exp_in_diff_sum = in_diff_sum*(1 / (3.1415*(1 + (in_diff_sum**2)))).cuda()
        exp_lat_diff_sum = lat_diff_sum*(1 / (3.1415*(1 + (lat_diff_sum**2)))).cuda()
        return output, exp_in_diff_sum, exp_lat_diff_sum, lat_repr#output, in_diff_sum, lat_diff_sum, lat_repr#