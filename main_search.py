# %%
from IPython import get_ipython
import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.datasets import make_s_curve
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from bayes_opt import BayesianOptimization
from radam import *
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from maguro import *
from model import *
#次元落とさずに3次元でできないか(1次元にせずチャンネルを保持)
#{'target': -1.0076094367541373, 'params': {'learning_rate': 0.0008826217021716755, 'wd': 0.0005360212814076004}}
#|  19       |  nan      |  0.000712 |  0.000520 |
do_this_bayes = {
    'learning_rate' : (1, 1000000),
    'wd' : (1, 1000000)
}

def custom_loss(output, target, in_diff_sum, lat_diff_sum):
    global g_distance, g_mse
    KL_divergence = nn.KLDivLoss(reduction="sum")
    SM = nn.Softmax(dim=0)
    g_mse = torch.mean(torch.sqrt((output - target)**2))
    g_distance = Variable(KL_divergence(SM(in_diff_sum).log(), SM(lat_diff_sum)), requires_grad = True).cuda()
    loss = g_mse+g_distance
    return loss


def z_score(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean) / xstd
    return zscore, xmean, xstd

def search_in_train(learning_rate, wd):
    model = autoencoder().cuda()
    criterion = custom_loss
    optimizer = RAdam(model.parameters(), lr=1/learning_rate, weight_decay=1/wd)#weight_decay
    in_tensor = torch.from_numpy(np_sr.astype(np.float32))
    for epoch in range(1, num_epochs+1):
        temp_loss = 0
        model.train()
        data_iter = DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True)
        for data in data_iter:
            batch = data
            batch = batch.reshape(batch.size(0)*3)
            batch = Variable(batch).cuda()
            # ===================forward=====================
            output, in_diff_sum, lat_diff_sum, _ = model(batch)
            loss = criterion(output, batch, in_diff_sum, lat_diff_sum)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_loss += loss.data.item() / (n_samples / BATCH_SIZE)
    return -(temp_loss)

n_samples = 2**10#32768
noise = 0.05
sr, color = make_swiss_roll(n_samples, noise)
np_sr = np.array(sr)
np_sr, input_mean, input_std = z_score(np_sr)

num_epochs = 20
early_stopping = 5
g_distance = torch.Tensor()
g_mse = torch.Tensor()

bayeser = BayesianOptimization(f=search_in_train, pbounds=do_this_bayes)
bayeser.maximize(init_points=200, n_iter=300)

#%%
print(bayeser.max)
