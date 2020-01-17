import torch
from torch import nn
import itertools
import numpy as np
BATCH_SIZE = 8#8
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
            nn.Dropout(),
            nn.Linear(C, B),
            Swish(),
            nn.Dropout(),
            nn.Linear(C, B),
            Swish(),
            nn.Linear(B, A),
            nn.Tanh())
    
    def forward(self, x):
        y = self.encoder(x)
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)#32, 2
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)#32,3
        sm=nn.Softmax(dim=0)
        in_data_sum = torch.sum(in_data, 1)#32,1
        lat_repr_sum = torch.sum(lat_repr, 1)#32,1
        distance = self.KL_divergence(sm(in_data_sum).log(), sm(lat_repr_sum))
        # print(in_data_sum.cpu().numpy())
        # print(in_data_sum.narrow(0, 1, len(in_data_sum)-1).cpu().numpy())
        # print(in_data_sum[0].reshape(1).cpu().numpy())
        # print(torch.cat([in_data_sum.narrow(0, 1, len(in_data_sum)-1), in_data_sum[0].reshape(1)], dim=0))
        in_diff = in_data_sum - torch.cat([in_data_sum.narrow(0, 1, len(in_data_sum)-1), in_data_sum[0].reshape(1)], dim=0)
        lat_diff = lat_repr_sum - torch.cat([lat_repr_sum.narrow(0, 1, len(lat_repr_sum)-1), lat_repr_sum[0].reshape(1)], dim=0)
        lipschitz = self.KL_divergence(sm(in_diff).log(), sm(lat_diff))
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
        return output, distance, lipschitz