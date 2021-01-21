import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
from torch import tan, atan
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from scipy.special import comb
BATCH_SIZE = 16#3#
INPUT_AXIS = 784
LATENT_DIMENSION = 2
shape_log = False
n_comb = comb(BATCH_SIZE, 2, exact=True)

def make_distance_tensor(input_tensor):
    input_diff_list = [torch.sqrt(torch.sum((input_tensor[n[0]]-input_tensor[n[1]])**2)) for n in itertools.combinations(range(BATCH_SIZE), 2)]
    input_diff_sum = torch.stack(input_diff_list, dim=0).cuda()
    return input_diff_sum

##########外側5%を省いて学習を繰り返す
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc1_1 = nn.Linear(400, 200)
        self.fc1_2 = nn.Linear(200, 50)
        self.fc2 = nn.Linear(50, 2)
        self.fc3 = nn.Linear(2, 50)
        self.fc3_1 = nn.Linear(50, 200)
        self.fc3_2 = nn.Linear(200, 400)
        self.fc4 = nn.Linear(400, 784)

    def encoder(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc1_1(x)
        x = F.leaky_relu(x)
        x = self.fc1_2(x)
        x = F.leaky_relu(x)
        return x

    def full_connection(self, x):
        x = self.fc2(x)
        return x

    def tr_full_connection(self, x):
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc3_1(x)
        x = F.leaky_relu(x)
        x = self.fc3_2(x)
        x = F.leaky_relu(x)
        return x

    def decoder(self, x):
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x
    
    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        x = x.view(-1, 784)
        z = self.encoder(x).cuda()
        if shape_log == True: print(f'z = encorder(x):{z.shape}')
        z = self.full_connection(F.tanh(z)).cuda()
        if shape_log == True: print(f'full_connection(z):{z.shape}')
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{z.shape}')
        y = self.decoder(y).view(BATCH_SIZE, 1, 28, 28).cuda()
        if shape_log == True: print(f'y:{y.shape}')
        #-----------------------------------------
        in_diff_sum = make_distance_tensor(x.reshape(BATCH_SIZE, INPUT_AXIS))
        lat_diff_sum = make_distance_tensor(z.reshape(BATCH_SIZE, LATENT_DIMENSION))
        out_diff_sum = make_distance_tensor(y.reshape(BATCH_SIZE, INPUT_AXIS))
        lat_repr = z.reshape(BATCH_SIZE, LATENT_DIMENSION).cuda()
        # in_diff_sum = torch.ones(n_comb)
        # lat_diff_sum = torch.ones(n_comb)
        # out_diff_sum = torch.ones(n_comb)
        #lat_repr = torch.ones(BATCH_SIZE, LATENT_DIMENSION)
        return y, in_diff_sum, lat_diff_sum, out_diff_sum, lat_repr