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
            nn.Linear(A, B),#96, 96
            Swish(),
            nn.Linear(B, C), #96, 96
            Swish(),
            nn.Linear(B, C), #96, 96
            Swish(),
            nn.Linear(C, D))#96,64
        self.decoder = nn.Sequential(
            nn.Linear(D, C),#64, 96
            Swish(),
            nn.Linear(C, B),
            Swish(),
            nn.Linear(C, B),
            Swish(),
            nn.Linear(B, A),
            nn.Tanh())
    
    def forward(self, x):
        y = self.encoder(x)
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)#32,3
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)#32, 2
        sm=nn.Softmax(dim=0)
        in_centroid = torch.sum(in_data, 0) / BATCH_SIZE#バッチごとの重心
        lat_centroid = torch.sum(lat_repr, 0) / BATCH_SIZE
        in_diff_list = []#全ての行から重心を引いて二乗
        for n in range(BATCH_SIZE):
            _sum = torch.sum((in_data[n]-in_centroid)**2)
            _sqrt = torch.sqrt(_sum)
            in_diff_list.append(_sqrt)
        in_diff_sum = torch.stack(in_diff_list, dim=0)
        lat_diff_list = []
        for n in range(BATCH_SIZE):
            _sum = torch.sum((lat_repr[n]-lat_centroid)**2)
            _sqrt = torch.sqrt(_sum)
            lat_diff_list.append(_sqrt)
        lat_diff_sum = torch.stack(lat_diff_list, dim=0)
        distance = self.KL_divergence(sm(in_diff_sum).log(), sm(lat_diff_sum))
        #########
        #lat_repr1, lat_repr2, lat_repr_sum1
        #    x          y          z
        #########
        #lat_repr_sumは、distanceが下がるほど正確になるため、
        #distanceがほぼ0のとき
        #[X, Y, Z]
        #[ak, bk, ck∝(X+Y+Z)] このときck∝(X+Y+Z)
        #[X, Y, Z]
        output = self.decoder(y)
        return output, distance