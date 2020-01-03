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
from sklearn.datasets import make_swiss_roll
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from bayes_opt import BayesianOptimization
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from maguro import *
from model import *
#次元落とさずに3次元でできないか(1次元にせずチャンネルを保持)

do_this_bayes = {
    'learning_rate' : (0.00001, 0.0001),
    'p1' : (0, 1),
    'p2' : (0, 1),
    'wd' : (0.000001, 0.0001)
}

def custom_loss(output, target, distance, lipschitz, p1, p2):
    global g_distance, g_mse
    g_mse = torch.mean((output - target)**2)
    g_distance = distance
    #loss = (g_mse+g_distance)*(1+torch.abs(g_distance-g_mse))+lipschitz#後ろを重視しすぎ
    loss = g_mse + (p1*(g_distance**2)) + (p2*(lipschitz**2))
    return loss

def z_score(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore, xmean, xstd

def search_in_train(learning_rate, p1, p2, wd):
    model = autoencoder().cuda()
    criterion = custom_loss#nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)#weight_decay

    in_tensor = torch.from_numpy(np_sr.astype(np.float32))

    all_loss=[]
    best_loss=99999
    es_count=0
    for epoch in range(1, num_epochs+1):
        for data in DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True):
            batch = data
            batch = batch.reshape(batch.size(0)*3)#考える必要あり
            batch = Variable(batch).cuda()
            # ===================forward=====================
            output, distance, lipschitz = model(batch)
            loss = criterion(output, batch, distance, lipschitz, p1, p2)
            #print(loss)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if loss.data.item() < best_loss:
            # print('[BEST] ', end='')
            # torch.save(model.state_dict(), f'./output/{epoch}.pth')
            best_loss = loss.data.item()
            es_count=0
        es_count += 1
        # print(f'epoch [{epoch}/{num_epochs}], loss:{loss.data.item()}, \
        #     \n g_mse={g_mse}, g_distance:{g_distance}, lipschitz:{lipschitz}'
        #     )
        all_loss.append(
            [epoch, loss.data.item(), g_mse.data.item(), \
            g_distance.data.item(), lipschitz.data.item()]
            )
        if es_count == early_stopping:
            # print('early stopping!')
            break#early_stopping
    return -(loss.data.item())

n_samples = 5096#32768
noise = 0#0.05
sr, color = make_swiss_roll(n_samples, noise)
np_sr = np.array(sr)
np_sr, input_mean, input_std = z_score(np_sr)

num_epochs = 200
early_stopping=10
g_distance = torch.Tensor()
g_mse = torch.Tensor()

optimizer = BayesianOptimization(f=search_in_train, pbounds=do_this_bayes)
optimizer.maximize(init_points=5, n_iter=5000)

#%%
print(optimizer.res['target'])
max_idx=np.argmax([x['target'] for x in optimizer.res])
print(optimizer.res[max_idx])
#targets=[for x in optimizer.res]
#print(target)
#{'target': -0.3682848811149597,
#  'params': {'learning_rate': 9.884352878854778e-05,
#  'p1': 0.9293929811106598,
#  'p2': 0.17363040237602712,
#  'wd': 5.358144068814221e-05}}
