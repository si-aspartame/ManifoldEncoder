import torch
from torch import nn
import itertools
import numpy as np
BATCH_SIZE = 32#8
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
            nn.Linear(A, B),#768, 128
            Swish(),
            nn.Linear(B, C), #64, 16
            Swish(),
            nn.Linear(B, C), #64, 16
            Swish(),
            nn.Linear(C, D))#16,4
        self.decoder = nn.Sequential(
            nn.Linear(D, C),
            Swish(),
            nn.Linear(C, B),
            Swish(),
            nn.Linear(C, B),
            Swish(),
            nn.Linear(C, B),
            Swish(),
            nn.Linear(B, A),
            nn.Tanh())
    
    def forward(self, x):
        y = self.encoder(x)
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)
        sm=nn.Softmax(dim=0)
        in_data_sum = sm(torch.sum(in_data, 1))#batch_size*1
        lat_repr_sum = sm(torch.sum(lat_repr, 1))#batch_size*1
        distance = self.KL_divergence(in_data_sum.log(), lat_repr_sum)
        out_tensor = self.decoder(y)#ここにin_data_sumかlat_repr_sumを追加してしまう
        
        return out_tensor, distance