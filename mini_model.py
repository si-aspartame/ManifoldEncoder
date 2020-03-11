import torch
from torch import nn
import itertools
import numpy as np
BATCH_SIZE = 8
INPUT_AXIS = 3
A = int(BATCH_SIZE*INPUT_AXIS)#1536
B = int(BATCH_SIZE*INPUT_AXIS)
C = int(BATCH_SIZE*INPUT_AXIS)
D = int(BATCH_SIZE*(INPUT_AXIS-1))#int(C/4)batch_size*(axis-1)

class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.KL_divergence = nn.KLDivLoss(reduction="sum")
        self.encoder = nn.Sequential(
            nn.Linear(A, B),#96, 96
            nn.ReLU(),
            nn.Linear(B, C), #96, 96
            nn.ReLU(),
            nn.Linear(C, C), #96, 96
            nn.ReLU(),
            nn.Linear(C, C), #96, 96
            nn.ReLU(),
            nn.Linear(C, C), #96, 96
            nn.ReLU(),
            nn.Linear(C, C), #96, 96
            nn.ReLU(),
            nn.Linear(C, D),
            Swish())#96,64
    
    def forward(self, x):
        y = self.encoder(x)
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)#32,3
        in_min = torch.min(in_data)
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)#32, 2
        lat_min  = torch.min(lat_repr)

        in_diff_list = []#各点同士の距離の組み合わせ
        for n, m in itertools.product(range(BATCH_SIZE), range(BATCH_SIZE)):
            in_cord1 = in_min + in_data[n]
            in_cord2 = in_min + in_data[m]
            in_diff_list.append(torch.sqrt(torch.sum((in_cord1-in_cord2)**2)))
        in_diff_sum = torch.stack(in_diff_list, dim=0)
        
        lat_diff_list = []
        for n, m in itertools.product(range(BATCH_SIZE), range(BATCH_SIZE)):
            lat_cord1 = lat_min + lat_repr[n]
            lat_cord2 = lat_min + lat_repr[m]
            lat_diff_list.append(torch.sqrt(torch.sum((lat_cord1-lat_cord2)**2)))
        lat_diff_sum = torch.stack(lat_diff_list, dim=0)

        #----------------------------------------
        return in_diff_sum, lat_diff_sum, lat_repr