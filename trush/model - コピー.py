import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
from torch import tan, atan
BATCH_SIZE = 4#3#
INPUT_AXIS = 3
A = int(BATCH_SIZE*INPUT_AXIS)#1536
B = int(BATCH_SIZE*INPUT_AXIS)
C = int(BATCH_SIZE*INPUT_AXIS)
D = int(BATCH_SIZE*(INPUT_AXIS-1))#int(C/4)batch_size*(axis-1)
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(A, B),####
#             nn.Linear(B, C),
            # nn.Linear(C, C),
            nn.Linear(C, D),####
            nn.Tanh(),
            )
        self.decoder = nn.Sequential(
            nn.Linear(D, C),####
            # nn.Linear(C, C),
#             nn.Linear(C, B),
            nn.Linear(B, A),####
            nn.Tanh()####Tanh
            )
    
    def forward(self, x):
        y = self.encoder(x).cuda()
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS).cuda()#32,3
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1).cuda()#32, 2
        in_min = torch.min(in_data).cuda()
        lat_min  = torch.min(lat_repr).cuda()
        #---------------------各点からの距離
        in_diff_list = []#各点同士の距離の組み合わせ
        for n in itertools.combinations(range(BATCH_SIZE), 2):#同じもの同士を比較しない
            in_cord1 = in_min + in_data[n[0]]
            in_cord2 = in_min + in_data[n[1]]
            #print(n, '|', in_cord1, '|', in_cord2)
            #print(torch.sum((in_cord1-in_cord2)**2))
            in_diff_list.append(torch.sqrt(torch.sum((in_cord1-in_cord2)**2)))
            #in_diff_list.append(torch.sum(in_cord1-in_cord2))
        in_diff_sum = torch.stack(in_diff_list, dim=0).cuda()
        #print('in_diff_sum:',in_diff_sum)

        lat_diff_list = []
        for n in itertools.combinations(range(BATCH_SIZE), 2):
            lat_cord1 = lat_min + lat_repr[n[0]]
            lat_cord2 = lat_min + lat_repr[n[1]]
            #print(n, '|', lat_cord1, '|', lat_cord2)
            lat_diff_list.append(torch.sqrt(torch.sum((lat_cord1-lat_cord2)**2)))
            #lat_diff_list.append(torch.sum(lat_cord1-lat_cord2))
        lat_diff_sum = torch.stack(lat_diff_list, dim=0).cuda()
        #print(lat_diff_sum)
        #print('lat_diff_sum:',in_diff_sum)


        #----------------------------------------
        output = self.decoder(y).cuda()
        # print(output)
        return output, in_diff_sum, lat_diff_sum, lat_repr