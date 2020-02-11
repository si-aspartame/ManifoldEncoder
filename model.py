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

to_positive = 2

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
            nn.Linear(B, C), #96, 96
            nn.ReLU(),
            nn.Linear(C, D))#96,64
        self.decoder = nn.Sequential(
            nn.Linear(D, C),#64, 96
            nn.ReLU(),
            nn.Linear(C, B),
            nn.ReLU(),
            nn.Linear(C, B), #96, 96
            nn.ReLU(),
            nn.Linear(B, A),
            nn.Tanh())
    
    def forward(self, x):
        y = self.encoder(x)
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)#32,3
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)#32, 2
        #入力と潜在表現の列方向の合計
        #この合計はマンハッタン距離と同値
        #ユークリッド距離 = マンハッタン距離*√2/2
        in_diff_list = []
        for n in range(BATCH_SIZE):
            in_sum = torch.sqrt(torch.sum((to_positive+in_data[n])**2))
            in_diff_list.append(in_sum)
        in_diff_sum = torch.stack(in_diff_list, dim=0)
        
        lat_diff_list = []
        for n in range(BATCH_SIZE):
            lat_sum = torch.sqrt(torch.sum((to_positive+lat_repr[n])**2))
            lat_diff_list.append(lat_sum)
        lat_diff_sum = torch.stack(lat_diff_list, dim=0)

        output = self.decoder(y)
        return output, in_diff_sum, lat_diff_sum, lat_repr