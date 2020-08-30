import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
N_LABELS = 3
N_CENTROIDS = 12
BATCH_SIZE = 1
class RadialBasisFunctionNetwork(nn.Module):
    def __init__(self):
        super(RadialBasisFunctionNetwork, self).__init__()
        self.Layer1 = nn.Linear(N_CENTROIDS, N_LABELS)#BATCH_SIZE -> BATCH_SIZE*N_CLUSTERS (supervised = one-hot)
    
    def forward(self, x):
        output = self.Layer1(x)
        return output