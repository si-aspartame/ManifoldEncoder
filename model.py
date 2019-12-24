import torch
from torch import nn
import itertools
import numpy as np
BATCH_SIZE = 512
INPUT_AXIS = 3
A = int(BATCH_SIZE*INPUT_AXIS)#1536
B = int(BATCH_SIZE*INPUT_AXIS)
C = int(BATCH_SIZE*INPUT_AXIS)
D = int(BATCH_SIZE*(INPUT_AXIS-1))#int(C/4)batch_size*(axis-1)

class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)

class autoencoder(nn.Module):
    # def __init__(self):
    #     super(autoencoder, self).__init__()
    #     self.encoder = nn.Sequential(
    #         nn.Linear(A, B),#768, 128
    #         nn.ReLU(True),
    #         nn.Linear(B, C), #64, 16
    #         nn.ReLU(True),
    #         nn.Linear(B, C), #64, 16
    #         nn.ReLU(True),
    #         nn.Linear(B, C), #64, 16
    #         nn.ReLU(True),
    #         nn.Linear(B, C), #64, 16
    #         nn.ReLU(True),
    #         nn.Linear(C, D))#16,4
    #     self.decoder = nn.Sequential(
    #         nn.Linear(D, C),
    #         nn.ReLU(True),
    #         nn.Linear(C, B),
    #         nn.ReLU(True),
    #         nn.Linear(C, B),
    #         nn.ReLU(True),
    #         nn.Linear(C, B),
    #         nn.ReLU(True),
    #         nn.Linear(C, B),
    #         nn.ReLU(True),
    #         nn.Linear(B, A),
    #         nn.Tanh())
    def __init__(self):
        super(autoencoder, self).__init__()
        self.KL_divergence = nn.KLDivLoss(reduction="sum")
        self.encoder = nn.Sequential(
            nn.Linear(A, B),#768, 128
            Swish(),
            nn.Linear(B, C), #64, 16
            Swish(),
            nn.Linear(C, D))#16,4
        self.decoder = nn.Sequential(
            nn.Linear(D, C),
            Swish(),
            nn.Linear(C, B),
            Swish(),
            nn.Linear(B, A),
            nn.Tanh())
        # self.encoder = nn.Sequential(
        #     nn.Linear(28 * 28, 128),#≒1/6
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),#1/2
        #     nn.ReLU(True), 
        #     nn.Linear(64, 12), #≒1/5
        #     nn.ReLU(True), 
        #     nn.Linear(12, 3))#1/4
        # self.decoder = nn.Sequential(
        #     nn.Linear(3, 12),
        #     nn.ReLU(True),
        #     nn.Linear(12, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 128),
        #     nn.ReLU(True), 
        #     nn.Linear(128, 28 * 28), 
        #     nn.Tanh())
    
    def forward(self, x):
        y = self.encoder(x)
        lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)
        in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)
        sm=nn.Softmax()
        in_data_sum = sm(torch.sum(in_data, 1))#batch_size*1
        lat_repr_sum = sm(torch.sum(lat_repr, 1))#batch_size*1
        #今は次の点との距離しか取得していないが、全ての点との距離の合計を取得する必要がある
        distance = self.KL_divergence(in_data_sum.log(), lat_repr_sum)
        out_tensor = self.decoder(y)
        return out_tensor, distance
    
    # def forward(self, x):#summation
    #     y = self.encoder(x)
    #     lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)
    #     in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)
    #     in_data_sum = torch.sum(in_data, 1).data.cpu().numpy()#batch_size*1
    #     lat_repr_sum = torch.sum(lat_repr, 1).data.cpu().numpy()#batch_size*1
    #     sum_of_diff=np.mean((in_data_sum-lat_repr_sum)**2)
    #     #print(sum_of_diff)#600前後
    #     out_tensor = self.decoder(y)
    #     return out_tensor, sum_of_diff

    # def forward(self, x):
    #     y = self.encoder(x)
    #     lat_repr = y.reshape(BATCH_SIZE, INPUT_AXIS-1)
    #     in_data = x.reshape(BATCH_SIZE, INPUT_AXIS)
    #     in_data_sum = torch.sum(in_data, 1).data.cpu().numpy()#batch_size*1
    #     lat_repr_sum = torch.sum(lat_repr, 1).data.cpu().numpy()#batch_size*1
    #     #今は次の点との距離しか取得していないが、全ての点との距離の合計を取得する必要がある
    #     d_diff = np.array([#入力の全ての点に対する3次元の合計
    #         s-sp1 for s, sp1 in \
    #         zip(in_data_sum, np.hstack([in_data_sum[1:], in_data_sum[0]]))])
    #     r_diff = np.array([
    #         s-sp1 for s, sp1 in \
    #         zip(lat_repr_sum, np.hstack([lat_repr_sum[1:], lat_repr_sum[0]]))])
    #     diff_ratio = d_diff / r_diff
    #     var_of_diff = np.var(diff_ratio)
    #     #print(var_of_diff)
    #     out_tensor = self.decoder(y)
    #     return out_tensor, var_of_diff

