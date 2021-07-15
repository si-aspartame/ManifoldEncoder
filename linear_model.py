import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from scipy.special import comb
BATCH_SIZE = 3
in_X = 28#1#
in_Y = 28#36#58#
INPUT_AXIS = in_X * in_Y

LATENT_DIMENSION = 2
shape_log = False
n_comb = comb(BATCH_SIZE, 2, exact=True)

def make_distance_vector(input_tensor):
    input_diff_list = [torch.sqrt(torch.sum((input_tensor[n[0]]-input_tensor[n[1]])**2)) for n in itertools.combinations(range(BATCH_SIZE), 2)]
    #input_diff_list = [torch.dot(input_tensor[n[0]], input_tensor[n[1]]) / (torch.linalg.norm(input_tensor[n[0]]) * torch.linalg.norm(input_tensor[n[1]])) for n in itertools.combinations(range(BATCH_SIZE), 2)]
    input_diff_sum = torch.stack(input_diff_list, dim=0).cuda()
    return input_diff_sum

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.fc1_1 = nn.Linear(INPUT_AXIS, 400)
        self.fc1_2 = nn.Linear(400, 200)
        self.fc1_3 = nn.Linear(200, 50)
        self.fc2   = nn.Linear(50, LATENT_DIMENSION)
        self.fc3   = nn.Linear(LATENT_DIMENSION, 50)
        self.fc4_1 = nn.Linear(50, 200)
        self.fc4_2 = nn.Linear(200, 400)
        self.fc4_3 = nn.Linear(400, INPUT_AXIS)
        # self.fc1_1 = nn.Linear(INPUT_AXIS, 28)
        # self.fc1_2 = nn.Linear(28, 16)
        # self.fc1_3 = nn.Linear(16, 8)
        # self.fc2   = nn.Linear(8, LATENT_DIMENSION)
        # self.fc3   = nn.Linear(LATENT_DIMENSION, 8)
        # self.fc4_1 = nn.Linear(8, 16)
        # self.fc4_2 = nn.Linear(16, 28)
        # self.fc4_3 = nn.Linear(28, INPUT_AXIS)
        # self.fc1_1 = nn.Linear(INPUT_AXIS, 16)
        # self.fc1_2 = nn.Linear(16, 8)
        # self.fc2   = nn.Linear(8, LATENT_DIMENSION)
        # self.fc3   = nn.Linear(LATENT_DIMENSION, 8)
        # self.fc4_2 = nn.Linear(8, 16)
        # self.fc4_3 = nn.Linear(16, INPUT_AXIS)

    def encoder(self, x):
        x = self.fc1_1(x)
        x = F.leaky_relu(x)
        x = self.fc1_2(x)
        x = F.leaky_relu(x)
        x = self.fc1_3(x)
        x = F.leaky_relu(x)
        return x

    def full_connection(self, z):
        z = self.fc2(z)
        return z

    def tr_full_connection(self, z):
        y = self.fc3(z)
        y = F.leaky_relu(y)
        return y

    def decoder(self, y):
        y = self.fc4_1(y)
        y = F.leaky_relu(y)
        y = self.fc4_2(y)
        y = F.leaky_relu(y)
        y = self.fc4_3(y)
        y = torch.tanh(y)
        return y
    
    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        x = x.view(-1, INPUT_AXIS)
        z = self.encoder(x).cuda()
        if shape_log == True: print(f'z = encorder(x):{x.shape}')
        z = self.full_connection(torch.tanh(z)).cuda()
        if shape_log == True: print(f'full_connection(z):{z.shape}')
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{z.shape}')
        y = self.decoder(y).view(BATCH_SIZE, 1, in_X, in_Y).cuda()
        if shape_log == True: print(f'y:{y.shape}')
        #-----------------------------------------
        in_diff_sum = make_distance_vector(x.reshape(BATCH_SIZE, INPUT_AXIS))
        lat_diff_sum = make_distance_vector(z.reshape(BATCH_SIZE, LATENT_DIMENSION))
        out_diff_sum = make_distance_vector(y.reshape(BATCH_SIZE, INPUT_AXIS))
        lat_repr = z.reshape(BATCH_SIZE, LATENT_DIMENSION).cuda()
        return y, in_diff_sum, lat_diff_sum, out_diff_sum, lat_repr