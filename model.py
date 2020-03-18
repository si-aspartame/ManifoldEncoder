import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
from torch import tan, atan
BATCH_SIZE = 4
INPUT_AXIS = 3
A = int(BATCH_SIZE*INPUT_AXIS)#1536
B = int(BATCH_SIZE*INPUT_AXIS)
C = int(BATCH_SIZE*INPUT_AXIS)
D = int(BATCH_SIZE*(INPUT_AXIS-1))#int(C/4)batch_size*(axis-1)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(A, B),#96, 96
            nn.Tanh(),
            nn.Linear(B, C),#96, 96
            nn.Tanh(),
            nn.Linear(C, C),
            nn.Linear(C, D),
            nn.Tanhshrink())#96,64 
        self.decoder = nn.Sequential(
            nn.Linear(D, C),#64, 96
            nn.Tanh(),
            nn.Linear(C, B),#64, 96
            nn.Tanh(),
            nn.Linear(B, B),
            nn.Linear(B, A),#64, 96
            nn.Tanh())
    
    def forward(self, x):
        y = self.encoder(x).cuda()
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS).cuda()#32,3
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1).cuda()#32, 2
        in_min = torch.min(in_data).cuda()
        lat_min  = torch.min(lat_repr).cuda()
        #---------------------各点からの距離
        in_diff_list = []#各点同士の距離の組み合わせ
        for n, m in itertools.product(range(BATCH_SIZE), range(BATCH_SIZE)):
            in_cord1 = in_min + in_data[n]
            in_cord2 = in_min + in_data[m]
            in_diff_list.append(torch.sqrt(torch.sum((in_cord1-in_cord2)**2)))
        in_diff_sum = Variable(torch.stack(in_diff_list, dim=0), requires_grad = True).cuda()
        
        lat_diff_list = []
        for n, m in itertools.product(range(BATCH_SIZE), range(BATCH_SIZE)):
            lat_cord1 = lat_min + lat_repr[n]
            lat_cord2 = lat_min + lat_repr[m]
            lat_diff_list.append(torch.sqrt(torch.sum((lat_cord1-lat_cord2)**2)))
        lat_diff_sum = Variable(torch.stack(lat_diff_list, dim=0), requires_grad = True).cuda()

        #----------------------------------------
        output = self.decoder(y).cuda()
        return output, in_diff_sum, lat_diff_sum, lat_repr